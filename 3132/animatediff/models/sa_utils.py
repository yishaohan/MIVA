import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, List, Optional, Tuple, Union

from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm, Attention

from einops import rearrange, repeat
import pdb
import numpy as np
from .lora import LoRALinearLayer
import math
from .pe import PositionalEncoding, PositionalEncoding_Rotary
from PIL import Image

# CFA layer implementation
class SparseCausalAttentionProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, 
        hidden_size=None, 
        use_q_lora: bool = False, 
        use_out_lora: bool = False, 
        lora_rank:int = None, # default rank=32, scale=1.0; for multi-adapter setting, the two should be lists.
        lora_scale:float = None,
        sa_pe_mode:str = "",
        zero_out_first_frame:bool = False,
        video_length:int = None,
        sa_mode:str = "first",
        sa_first_frame_scale = 0.5,
        use_mask_branch: bool = False,
        mask_branch_kwargs: dict = {},
        ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.use_q_lora = use_q_lora
        self.use_out_lora = use_out_lora
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        self.video_length = video_length
        self.temporal_position_encoding_max_len = self.video_length 
        self.sa_pe_mode = sa_pe_mode
        self.sa_mode = sa_mode
        self.sa_first_frame_scale = sa_first_frame_scale
        self.use_mask_branch = use_mask_branch
        self.use_image_guidance = mask_branch_kwargs.get("use_image_guidance", False)
        if sa_pe_mode:
            PE_class = PositionalEncoding_Rotary
            self.pos_encoder = PE_class(
                    hidden_size,
                    dropout=0., 
                    max_len=self.temporal_position_encoding_max_len,
                    mode=sa_pe_mode,
                )
            print("PE for Q:", sa_pe_mode)
        else:
            self.pos_encoder = None
        self.zero_out_first_frame = zero_out_first_frame
        if self.use_q_lora: 
            self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)
        if self.use_out_lora:
            self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)

        if self.use_mask_branch:
            self.to_q_mask = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_mask = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Dropout(0.1)
            )

        self.use_weight_module = isinstance(sa_first_frame_scale, str) and 'adaptive' in sa_first_frame_scale
        if self.use_weight_module:
            temperature = 1 # adaptive_{number}, number being the temperature, otherwise 1
            if isinstance(sa_first_frame_scale, str):
                words = self.sa_first_frame_scale.split('_')
                if len(words)>1:
                    temperature = float(words[-1])
            self.weight_module = AdaptiveWeight(temperature=temperature)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        use_mask_branch = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        video_length = self.video_length

        if use_mask_branch is None:
            use_mask_branch = self.use_mask_branch
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            # assume 16:10 ratio
            height = int(math.sqrt(hidden_states.shape[1]/1.6))
            width = int(1.6*height)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if self.sa_pe_mode:
            hidden_states = rearrange(hidden_states,"b c f h w -> b f h w c")
            hidden_states = self.pos_encoder(hidden_states)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if True else (scale,) # if USE_PEFT_BACKEND else (scale,)

        if use_mask_branch:
            if not batch_size == video_length * 2:
                raise ValueError("expected latent with video and mask sequences, found incorrect sequence length!")
            mask_states = hidden_states[video_length:, ...]
            hidden_states = hidden_states[:video_length, ...]
            batch_size = batch_size // 2

        if attention_mask is not None:
            bsz = batch_size
            if "first" in self.sa_mode and "prev" in self.sa_mode:
                attention_mask = rearrange(attention_mask, "b h c (m d) -> (b m) h c d", m=2)
                bsz *= 2
            attention_mask = downscale(attention_mask, height, width)
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, bsz)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(bsz, attn.heads, -1, attention_mask.shape[-1])
            if "first" in self.sa_mode and "prev" in self.sa_mode:
                attention_mask = rearrange(attention_mask, "(b m) h c d -> b h c (m d)", m=2)

        query = attn.to_q(hidden_states, *args) 
        if self.use_q_lora: 
            query += self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if use_mask_branch:
            query_m = self.to_q_mask(mask_states, *args)
            key_m = attn.to_k(mask_states, *args)
            value_m = attn.to_v(mask_states, *args)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        # SparseCausalAttn

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key_c = key.clone()
        if "first" in self.sa_mode and "prev" in self.sa_mode: # "first_prev"
            key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        elif "first" in self.sa_mode:
            key = key[:,  [0] * video_length]
        elif "prev" in self.sa_mode:
            key = key[:, former_frame_index]
        key = rearrange(key, "b f d c -> (b f) d c")
        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value_c = value.clone()
        if "first" in self.sa_mode and "prev" in self.sa_mode: # "first_prev"
            if type(self.sa_first_frame_scale) == float:
                value = torch.cat([value[:, [0] * video_length] * self.sa_first_frame_scale, value[:, former_frame_index] * (1-self.sa_first_frame_scale)], dim=2)
            else:
                if self.sa_first_frame_scale == "linear":
                    weight = (timestep.cpu().item()) / 1000
                elif self.use_weight_module:
                    # getting weight from Adaptive Weight Module
                    weight = self.weight_module(temb)[0,0]
                value = torch.cat([value[:, [0] * video_length] * weight, value[:, former_frame_index] * (1-weight)], dim=2)

        elif "first" in self.sa_mode:
            value = value[:,  [0] * video_length]
        elif "prev" in self.sa_mode:
            value = value[:, former_frame_index]
        value = rearrange(value, "b f d c -> (b f) d c")

        if use_mask_branch:
            key_m = rearrange(key_m, "(b f) d c -> b f d c", f=video_length)
            if "first" in self.sa_mode and "prev" in self.sa_mode: # "first_prev"
                key_m = torch.cat([key_m[:, [0] * video_length], key_m[:, former_frame_index]], dim=2)
            elif "first" in self.sa_mode:
                key_m = key_m[:,  [0] * video_length]
            elif "prev" in self.sa_mode:
                key_m = key_m[:, former_frame_index]
            
            if self.use_image_guidance:
                key_m = torch.cat([key_m, key_c], dim=2)
            key_m = rearrange(key_m, "b f d c -> (b f) d c")

            value_m = rearrange(value_m, "(b f) d c -> b f d c", f=video_length)
            if "first" in self.sa_mode and "prev" in self.sa_mode: # "first_prev"
                if type(self.sa_first_frame_scale) == float:
                    value_m = torch.cat([value_m[:, [0] * video_length] * self.sa_first_frame_scale, value_m[:, former_frame_index] * (1-self.sa_first_frame_scale)], dim=2)
                else:
                    if self.sa_first_frame_scale == "linear":
                        weight = (timestep.cpu().item()) / 1000
                    elif self.use_weight_module:
                        weight = self.weight_module(temb)[0,0]
                    value_m = torch.cat([value_m[:, [0] * video_length] * weight, value_m[:, former_frame_index] * (1-weight)], dim=2)
            elif "first" in self.sa_mode:
                value_m = value_m[:,  [0] * video_length]
            elif "prev" in self.sa_mode:
                value_m = value_m[:, former_frame_index]
            
            if self.use_image_guidance:
                value_m = torch.cat([value_m, value_c], dim=2)
            value_m = rearrange(value_m, "b f d c -> (b f) d c")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # SparseCausalAttn

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        if use_mask_branch:
            query_m = query_m.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key_m = key_m.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_m = value_m.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            mask_states = F.scaled_dot_product_attention(
                query_m, key_m, value_m, attn_mask=None, dropout_p=0.0, is_causal=False
            )        
        

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        if self.use_out_lora:
            hidden_states += self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if use_mask_branch:
            mask_states = mask_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            mask_states = mask_states.to(query.dtype)

            mask_states = self.to_out_mask[0](mask_states, *args)
            mask_states = self.to_out_mask[1](mask_states)
            
            # merge video latent & mask latent
            hidden_states = torch.concat([hidden_states, mask_states], dim=0)
            batch_size *= 2

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.zero_out_first_frame:
            frame_len = video_length
            if use_mask_branch:
                frame_len *= 2
            hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", f=frame_len)
            hidden_states[:,0,:,:] = 0
            hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c", f=frame_len)

        return hidden_states

# downsize attn_mask to appropriate resolution
def downscale(attn_mask, height, width):
    b, head, hw_old, _ = attn_mask.shape
    r_square = hw_old // (height * width)
    r = int(math.sqrt(r_square))
    h_old = height * r
    w_old = width * r

    attn_mask = attn_mask.reshape(b, head, h_old, w_old, h_old, w_old)
    attn_mask = attn_mask.reshape(-1, 1, h_old, w_old)
    attn_mask = F.interpolate(attn_mask, scale_factor=1/r, mode="area")

    attn_mask = attn_mask.reshape(b, head, h_old, w_old, height, width)
    attn_mask = attn_mask.permute(0, 1, 4, 5, 2, 3)
    attn_mask = attn_mask.reshape(-1, 1, h_old, w_old)
    attn_mask = F.interpolate(attn_mask, scale_factor=1/r, mode="area")
    
    attn_mask = attn_mask.reshape(b, head, height, width, height, width)
    attn_mask = attn_mask.permute(0, 1, 4, 5, 2, 3).contiguous()
    attn_mask = attn_mask.reshape(b, head, height * width, -1)

    return attn_mask

# deprecated
class LoRASparseCausalAttentionProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size=None,
        cross_attention_dim=None,
        lora_rank:int=32,
        lora_scale:float=1.0):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        if self.lora_rank is not None and self.lora_scale is not None:
            self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        video_length = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if True else (scale,) # if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states,  *args) if self.lora_rank is None and self.lora_scale is None else self.to_q_lora(hidden_states) + attn.to_q(hidden_states, *args)


        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args) if self.lora_rank is None and self.lora_scale is None else self.to_k_lora(hidden_states) + attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args) if self.lora_rank is None and self.lora_scale is None else self.to_v_lora(hidden_states) + attn.to_v(encoder_hidden_states, *args) 
        

        # SparseCausalAttn
        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = key[:,  [0] * video_length]
        key = rearrange(key, "b f d c -> (b f) d c")
        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = value[:,  [0] * video_length]
        value = rearrange(value, "b f d c -> (b f) d c")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # SparseCausalAttn
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args) if self.lora_rank is None and self.lora_scale is None else self.to_out_lora(hidden_states) + attn.to_out[0](hidden_states, *args)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
# Adaptive Weight Module Implementation
class AdaptiveWeight(nn.Module):
    r"""
    Originally from diffusers.models.normalization.AdaLayerNormZero
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary. In our codebase it's 1280.
    """

    def __init__(self, embedding_dim: int=1280, bias=True, temperature=1):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2, bias=bias)
        self.temperature = temperature

    def forward(
        self,
        emb: Optional[torch.Tensor] = None, # the time emb
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))

        weight = torch.softmax(emb / self.temperature, dim=-1)
        
        return weight
    
    def visualize(self, unet, ckpt_path:str, key:str):
        '''
        output the weights corresponding to t=0,1,...,999.
        '''
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps
        time_proj = Timesteps(320, True, 0)
        time_embedding = unet.time_embedding

        timesteps = torch.tensor([i for i in range(1000)], dtype=torch.float32, device=self.linear.weight.device)

        t_emb = time_proj(timesteps).to(dtype=torch.float32)
        emb = time_embedding(t_emb)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            key_w = key + '.linear.weight'
            key_b = key + '.linear.bias'

            self.linear.weight.requires_grad = False
            self.linear.weight.copy_(ckpt[key_w])

            self.linear.bias.requires_grad = False
            self.linear.bias.copy_(ckpt[key_b])

        weights = self.forward(emb)

        return weights