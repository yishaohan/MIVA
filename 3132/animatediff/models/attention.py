# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import math
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm, Attention
from einops import rearrange, repeat
import pdb

from .sa_utils import SparseCausalAttentionProcessor, downscale
from .ca_utils import LoRACAAttentionProcessor
from .t_ca_utils import TemporalCAAttentionProcessor

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_motion_embedding = False,
        use_attn_temp_lora = True,
        use_i2v = True,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_temporal_CA=None,

        num_adapters: int = 1,
        adapter_weights = [1], 
        use_i2v_q_lora = False, 
        use_i2v_out_lora = False,
        use_mask_branch = False,
        mask_branch_kwargs = {},
        i2v_lora_rank = None,
        i2v_lora_scale = None,
        use_ca_lora = False,
        ca_lora_rank = None,
        ca_lora_scale = None,

        q_downsample:bool = False,
        q_downsample_ratio:int = 4,
        ca_pe_mode:str = None,
        use_CA_att_mask_framewise:bool = False,
        use_simple_CA:bool = False,
        simple_CA_kwargs: dict = {},
        sa_mode:str = "first",
        sa_first_frame_scale:float = 0.5,
        use_q_lora:bool = False,
        use_ip_adapter:str = None,

        remove_WO: bool = False,
        video_length:int = 16,
        parallel_mode: str = 'weights',
        zero_out_first_frame:bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.use_mask_branch = use_mask_branch
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_motion_embedding = use_motion_embedding,
                    use_attn_temp_lora = use_attn_temp_lora,
                    use_i2v = use_i2v,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                    use_temporal_CA=use_temporal_CA,
                    
                    num_adapters=num_adapters,
                    adapter_weights = adapter_weights, 
                    use_i2v_q_lora = use_i2v_q_lora, 
                    use_i2v_out_lora = use_i2v_out_lora,
                    use_mask_branch = use_mask_branch,
                    mask_branch_kwargs = mask_branch_kwargs,
                    i2v_lora_rank = i2v_lora_rank,
                    i2v_lora_scale = i2v_lora_scale,
                    ca_lora_rank=ca_lora_rank,
                    ca_lora_scale=ca_lora_scale,
                    use_ca_lora=use_ca_lora,
                    q_downsample=q_downsample,
                    q_downsample_ratio=q_downsample_ratio,
                    ca_pe_mode=ca_pe_mode,
                    use_CA_att_mask_framewise=use_CA_att_mask_framewise,
                    use_simple_CA=use_simple_CA,
                    simple_CA_kwargs = simple_CA_kwargs,
                    sa_mode=sa_mode,
                    sa_first_frame_scale=sa_first_frame_scale,
                    use_q_lora=use_q_lora,
                    use_ip_adapter=use_ip_adapter,
                    remove_WO=remove_WO,
                    video_length=video_length,
                    parallel_mode=parallel_mode,
                    zero_out_first_frame=zero_out_first_frame,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_masks={}, use_mask_branch=None, return_dict: bool = True, attention_store = None, tCA_attention_store = None, location = None, temb=None, ):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        if use_mask_branch is None:
            use_mask_branch = self.use_mask_branch
        if use_mask_branch:
            video_length = video_length // (np.sum(self.use_mask_branch) + 1)
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        # "C U" => "CCCC UUUU" for 1st dim (f=4)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_masks=attention_masks,
                timestep=timestep,
                video_length=video_length, 
                attention_store = attention_store,
                tCA_attention_store = tCA_attention_store,
                location = location,
                temb=temb,
                use_mask_branch=use_mask_branch,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length * (np.sum(self.use_mask_branch) + 1) if use_mask_branch else video_length)
        if not return_dict:
            return (output,)
            
        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,

        unet_use_cross_frame_attention = None,
        unet_use_temporal_attention = None,
        use_temporal_CA = None,
        use_motion_embedding = True,
        use_attn_temp_lora= True,
        use_i2v = True,
        lora_rank = 64,
        lora_scale = 1.0,
        parallel_mode = 'weights',

        num_adapters = 1,
        adapter_weights = [1],
        use_i2v_q_lora = False, 
        use_i2v_out_lora = False,
        use_mask_branch = False,
        mask_branch_kwargs = {},
        i2v_lora_rank = None,
        i2v_lora_scale = None,
        use_ca_lora = False,
        ca_lora_rank = None,
        ca_lora_scale = None,

        q_downsample:bool = False,
        q_downsample_ratio:int = 4,
        ca_pe_mode:str = None,
        use_CA_att_mask_framewise:bool = False,
        use_simple_CA:bool = False,
        simple_CA_kwargs:dict = {},
        sa_mode:str = "first",
        sa_first_frame_scale:float = 0.5,
        use_q_lora:bool = False,
        use_ip_adapter:str = None,

        remove_WO: bool = False,
        video_length:int = 16,

        zero_out_first_frame: bool = False,
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.use_temporal_CA = use_temporal_CA
        self.scale = 1.0
        self.adapter_weights = adapter_weights
        self.zero_out_first_frame = zero_out_first_frame
        self.video_length = video_length
        self.sa_mode = sa_mode
        self.use_mask_branch = use_mask_branch

        # SC-Attn
        assert unet_use_cross_frame_attention is not None
        if unet_use_cross_frame_attention:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = CrossAttention( # this is actually a SA
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        self.num_i2v_adapters = num_adapters
        # CFA layer
        if num_adapters == 1:
            self.i2v_adapter = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                processor = SparseCausalAttentionProcessor(hidden_size = dim,
                                                           use_q_lora = use_i2v_q_lora,
                                                           use_out_lora = use_i2v_out_lora,
                                                           lora_rank = i2v_lora_rank,
                                                           lora_scale = i2v_lora_scale,
                                                           zero_out_first_frame = self.zero_out_first_frame,
                                                           video_length=video_length,
                                                           sa_mode=sa_mode,
                                                           sa_first_frame_scale=sa_first_frame_scale,
                                                           use_mask_branch=use_mask_branch,
                                                           mask_branch_kwargs=mask_branch_kwargs) 
            )
        elif num_adapters > 1:
            i2v_adapters = [
                Attention(
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                    upcast_attention=upcast_attention,
                    processor = SparseCausalAttentionProcessor(hidden_size = dim,
                                                           use_q_lora = use_i2v_q_lora,
                                                           use_out_lora = use_i2v_out_lora,
                                                           lora_rank = i2v_lora_rank,
                                                           lora_scale = i2v_lora_scale,
                                                           zero_out_first_frame = self.zero_out_first_frame,
                                                           video_length=video_length,
                                                           sa_mode=sa_mode,
                                                           sa_first_frame_scale=sa_first_frame_scale,
                                                           use_mask_branch=use_mask_branch[i],
                                                           mask_branch_kwargs=mask_branch_kwargs) 
                ) for i in range(num_adapters)
            ]
            self.i2v_adapters = nn.ModuleList(i2v_adapters)

        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn with implicit prompt
        self.use_simple_CA = use_simple_CA
        self.CA_mask_guidance = "CA" in mask_branch_kwargs.get("mask_guided_attention", "SA")
        if use_simple_CA:
            self.sCA = Simple_CA(dim=dim, bias=attention_bias, num_adapters=num_adapters, **simple_CA_kwargs)
        if cross_attention_dim is not None:
            if use_ca_lora:
                lora_rank = ca_lora_rank
                lora_scale = ca_lora_scale
            else:
                lora_rank = None
                lora_scale = None
            

            self.attn2 = CrossAttention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    processor = LoRACAAttentionProcessor(
                        hidden_size=dim,
                        cross_attention_dim=cross_attention_dim,
                        num_adapters=num_adapters,
                        adapter_weights=adapter_weights,
                        lora_rank=lora_rank, lora_scale=lora_scale,
                        q_downsample=q_downsample,
                        q_downsample_ratio=q_downsample_ratio,
                        ca_pe_mode=ca_pe_mode,
                        use_CA_att_mask_framewise=use_CA_att_mask_framewise,
                        use_q_lora=use_q_lora,
                        use_ip_adapter=use_ip_adapter,
                        remove_WO=remove_WO,
                        video_length=video_length,
                        parallel_mode=parallel_mode,
                        zero_out_first_frame=zero_out_first_frame,
                    ) 
               )
           
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        assert use_temporal_CA is not None
        if (use_temporal_CA):
            self.attn_tCA = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                processor = TemporalCAAttentionProcessor(
                        cross_attention_dim=cross_attention_dim,
                        inner_dim=num_attention_heads * attention_head_dim, # dim_head * heads
                        video_length = video_length,
                    ) 
            )
            self.tCA_norm = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn_temp.set_use_memory_efficient_attention_xformers(False)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_masks={}, video_length=None, attention_store=None, tCA_attention_store=None, location = None, temb=None, use_mask_branch=None):
        # SparseCausal-Attention
        if use_mask_branch is None:
            use_mask_branch = self.use_mask_branch

        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        # Run I2V adapter
        i2v_hidden_states = torch.zeros_like(norm_hidden_states)
        if self.num_i2v_adapters == 1:
            i2v_hidden_states = self.i2v_adapter(norm_hidden_states, attention_mask=attention_masks.get("SA", None), timestep=timestep, temb=temb, use_mask_branch=use_mask_branch)
        elif self.num_i2v_adapters > 1:
            for index, i2v_adapter in enumerate(self.i2v_adapters):
                if self.use_mask_branch[index]:
                    i2v_hidden_state = i2v_adapter(torch.concat([norm_hidden_states[:video_length, ...], norm_hidden_states[(index+1) * video_length:(index+2) * video_length, ...]]), 
                    attention_mask=attention_masks.get("SA", [None] * (index + 1))[index], timestep=timestep, temb=temb, use_mask_branch=use_mask_branch) * self.adapter_weights[index]
                    i2v_hidden_states[:video_length] += i2v_hidden_state[:video_length]
                    i2v_hidden_states[(index+1) * video_length:(index+2) * video_length] += i2v_hidden_state[video_length:]
                else:
                    i2v_hidden_states[:video_length] += i2v_adapter(norm_hidden_states[:video_length, ...], timestep=timestep, temb=temb, use_mask_branch=use_mask_branch) * self.adapter_weights[index]

        if self.unet_use_cross_frame_attention: # default=False
            hidden_states = self.attn1(norm_hidden_states,encoder_hidden_states, attention_mask=None) + i2v_hidden_states + hidden_states
        else:
            # should run here
            SA_original_mask = attention_masks.get("SA_original", None)
            if SA_original_mask is not None:
                height = int(math.sqrt(hidden_states.shape[1]/1.6))
                width = int(1.6*height)
                SA_original_mask = downscale(SA_original_mask, height, width)
                attention_mask_m = torch.zeros_like(SA_original_mask)
                SA_original_mask = torch.stack([SA_original_mask, attention_mask_m])

            hidden_states_attn1 = self.attn1(norm_hidden_states, attention_mask=SA_original_mask)
            if self.zero_out_first_frame:
                hidden_states_attn1 = self.zero_out_CrossAttention(hidden_states_attn1)

            if self.sa_mode == 'first_naive':
                hidden_states = hidden_states_attn1 * 0.5 + i2v_hidden_states * 0.5 + hidden_states
            else:
                hidden_states = self.scale * hidden_states_attn1 + i2v_hidden_states + hidden_states
        if self.use_simple_CA:
            if self.sCA.simple_ca_norm:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
            else: 
                norm_hidden_states = hidden_states
            CA_mask = attention_masks.get("CA", None)
            hidden_states = (
                self.sCA(
                    norm_hidden_states, num_adapters=self.num_adapters, adapter_weights=self.adapter_weights, attention_store=attention_store, location=location, attention_mask=CA_mask, video_length=video_length
                )
                + hidden_states
            )
        elif self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            if use_mask_branch:
                encoder_hidden_states = torch.concat([encoder_hidden_states, encoder_hidden_states], dim=0)
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=None, attention_store=attention_store, location=location,
                )
                + hidden_states
            )
        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        if self.use_temporal_CA:
            norm_hidden_states = (
                self.tCA_norm(hidden_states, timestep) if self.use_ada_layer_norm else self.tCA_norm(hidden_states)
            )
            hidden_states = (
                self.attn_tCA(norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=None, tCA_attention_store=tCA_attention_store, location=location)
                + hidden_states
            )

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            if use_mask_branch:
                m = np.sum(self.use_mask_branch) + 1
                hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length * m)
                hidden_states = rearrange(hidden_states, "b (m f) c -> (b m) f c", m=m)
            else:
                hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            if use_mask_branch:
                hidden_states = rearrange(hidden_states, "(b m) f c -> b (m f) c", m=m)
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
            else:
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
    
    def zero_out_CrossAttention(self, hidden_states):
        hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", f=self.video_length)
        hidden_states[:,0,:,:] = 0
        hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c")

        return hidden_states


# CA with implicit prompt
class Simple_CA(nn.Module):
    def __init__(self, dim, bias=False,
                 num_adapters=1, rank=64, norm=False, act="gelu", norm_mid="layernorm", 
                 out=False):
        super().__init__()

        self.CA1 = nn.Linear(dim, dim, bias)
        self.simple_ca_norm = norm
        if act == "gelu":
            self.activation_fn = F.gelu
        else:
            self.activation_fn = nn.Identity

        if norm_mid == "layernorm":
            if num_adapters > 1:
                self.CA_norm = nn.ModuleList([nn.LayerNorm(rank) for _ in range(num_adapters)])
            else:
                self.CA_norm = nn.LayerNorm(rank)
        else:
            if num_adapters > 1:
                self.CA_norm = nn.ModuleList([nn.Identity for _ in range(num_adapters)])
            else:
                self.CA_norm = nn.Identity

        self.use_out = out
        if num_adapters > 1:
            self.CA2 = nn.ModuleList([nn.Linear(dim, rank, bias=bias) for _ in range(num_adapters)])
            self.CA3 = nn.ModuleList([nn.Linear(rank, dim, bias=bias) for _ in range(num_adapters)])
        else:
            self.CA2 = nn.Linear(dim, rank, bias=bias)
            self.CA3 = nn.Linear(rank, dim, bias=bias)
        self.CA4 = nn.Linear(dim, dim, bias)

    def forward(self,
                hidden_states,
                num_adapters,
                adapter_weights=[],
                attention_store=None,
                location=None,
                attention_mask=None,
                video_length=16, 
                use_mask_branch = False,
                ):
        if num_adapters > 1:
            res = torch.zeros_like(hidden_states)
            mask_idx = 0
            for i in range(num_adapters):
                hs = hidden_states[:video_length]
                if attention_mask is not None and attention_mask[i] is not None:
                    hs = torch.concat([hs, hidden_states[:, mask_idx*video_length:(mask_idx+1)*video_length, :]])
                cur = self.CA3[i](self.activation_fn(self.CA_norm[i](self.CA2[i](self.CA1(hs)))))
                if self.use_out:
                    cur = self.CA4(cur)

                if attention_mask is not None and attention_mask[i] is not None:
                    mask = attention_mask[i]
                    mask = mask.unsqueeze(1) #(f, 1, h, w)
                    hw = hidden_states.shape[1]
                    r_square = (mask.shape[-2] * mask.shape[-1]) // hw
                    r = int(math.sqrt(r_square))
                    mask = F.interpolate(mask, scale_factor=1/r, mode="area")
                    mask = rearrange(mask, "f d h w -> f (h w) d") > 0.5

                    if use_mask_branch:
                        mask_m = torch.ones_like(mask)
                        mask = torch.cat([mask, mask_m], dim=0).expand(hidden_states.shape)
                    cur = cur * mask
                    res[mask_idx*video_length:(mask_idx+1)*video_length] += cur[video_length:] * adapter_weights[i]
                    mask_idx += 1
 
                res[:video_length] += cur[:video_length] * adapter_weights[i]
            hidden_states = res
        else:
            A = self.activation_fn(self.CA_norm(self.CA2(self.CA1(hidden_states))))
            if attention_store is not None:
                attention_store.store(A, True, location)
            if self.use_out:
                hidden_states = self.CA4(self.CA3(A))
            else:
                hidden_states = self.CA3(A)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1) #(f, 1, h, w)
            hw = hidden_states.shape[1]
            r_square = (attention_mask.shape[-2] * attention_mask.shape[-1]) // hw
            r = int(math.sqrt(r_square))
            attention_mask = F.interpolate(attention_mask, scale_factor=1/r, mode="area")
            attention_mask = rearrange(attention_mask, "f d h w -> f (h w) d") > 0.5

            if hidden_states.shape[0] == attention_mask.shape[0] * 2:
                attention_mask_m = torch.ones_like(attention_mask)
                attention_mask = torch.cat([attention_mask, attention_mask_m], dim=0).expand(hidden_states.shape)
            hidden_states = hidden_states * attention_mask
        return hidden_states