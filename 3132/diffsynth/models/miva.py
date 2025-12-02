import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Any, Callable, List, Optional, Tuple, Union
from diffsynth.models.wan_video_dit import AttentionModule, RMSNorm, sinusoidal_embedding_1d

# Adaptive Weight Module Implementation
class AdaptiveWeight(nn.Module):
    r"""
    Originally from diffusers.models.normalization.AdaLayerNormZero
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary. In our codebase it's 1536.
    """

    def __init__(self, embedding_dim: int=1536, bias=True, temperature=1):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2, bias=bias)
        self.temperature = temperature

    def forward(
        self,
        t_emb: Optional[torch.Tensor] = None, # the time emb
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_emb = self.linear(self.silu(t_emb))

        weight = torch.softmax(t_emb / self.temperature, dim=-1)
        
        return weight
    
    def visualize(self, dit, ckpt_path: str, key: str):

        device = self.linear.weight.device
        dtype = self.linear.weight.dtype

        timesteps = torch.arange(1000, dtype=dtype, device=device)

        t_emb = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timesteps)).to(dtype=dit.time_embedding[0].weight.dtype)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            key_w = key + '.linear.weight'
            key_b = key + '.linear.bias'

            self.linear.weight.requires_grad = False
            self.linear.weight.copy_(ckpt[key_w])

            self.linear.bias.requires_grad = False
            self.linear.bias.copy_(ckpt[key_b])

        weights = self.forward(t_emb)

        return weights

# CFA layer implementation
class I2VAdapter(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, SA_mode: str = 'first',
                 sa_first_frame_scale = 0.5, # can be str or float
                 use_mask_branch = False,
                 mask_branch_kwargs = {},
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.use_mask_branch = use_mask_branch
        self.mask_branch_kwargs = mask_branch_kwargs

        if use_mask_branch:
            self.q_m = nn.Linear(dim, dim)
            self.o_m = nn.Linear(dim, dim)

        self.attn = AttentionModule(self.num_heads)

        self.sa_mode = SA_mode
        self.sa_first_frame_scale = sa_first_frame_scale

        self.use_weight_module = (isinstance(sa_first_frame_scale, str) and 'adaptive' in sa_first_frame_scale)
        if self.use_weight_module:
            temperature = 1 # adaptive_{number}, number being the temperature, otherwise 1
            if isinstance(sa_first_frame_scale, str):
                words = self.sa_first_frame_scale.split('_')
                if len(words)>1:
                    temperature = float(words[-1])
            self.weight_module = AdaptiveWeight(temperature=temperature)

    def forward(self, x: torch.Tensor, grid_size, timestep, t_emb, attention_mask = None):
        f, h, w = grid_size
        num_patches_per_frame = h * w

        if self.use_mask_branch:
            # x = rearrange(x, "(m b) n c -> m b n c", m = 2)
            x, x_m = torch.chunk(x, chunks=2, dim=0)
            x_m = x_m.clone()

        ff_index = torch.arange(f) - 1
        ff_index[0] = 0

        q = self.norm_q(self.q(x))
        if self.use_mask_branch:
            q_m = self.norm_q(self.q_m(x_m))

        if "first" in self.sa_mode and "prev" in self.sa_mode: # "first_prev"
            x = rearrange(x, 'b (f n) c -> (b f) n c', f=f, n=num_patches_per_frame)
            k = self.norm_k(self.k(x))
            v = self.v(x)

            k = rearrange(k, '(b f) n c -> b f n c', f=f, n=num_patches_per_frame)
            v = rearrange(v, '(b f) n c -> b f n c', f=f, n=num_patches_per_frame)

            k = torch.cat([k[:, [0] * f], k[:, ff_index]], dim=2)

            if type(self.sa_first_frame_scale) == float:
                v = torch.cat([v[:, [0] * f] * self.sa_first_frame_scale, v[:, ff_index] * (1-self.sa_first_frame_scale)], dim=2)
            else:
                if self.sa_first_frame_scale == "linear":
                    weight = (timestep.cpu().item()) / 1000
                elif self.use_weight_module:
                    weight = self.weight_module(t_emb)[0,0]
                v = torch.cat([v[:, [0] * f] * weight, v[:, ff_index] * (1-weight)], dim=2)

            k = rearrange(k, 'b f n c -> (b f) n c', f=f)
            v = rearrange(v, 'b f n c -> (b f) n c', f=f)

            if self.use_mask_branch:
                x_m = rearrange(x_m, 'b (f n) c -> (b f) n c', f=f, n=num_patches_per_frame)
                k_m = self.norm_k(self.k(x_m))
                v_m = self.v(x_m)

                k_m = rearrange(k_m, '(b f) n c -> b f n c', f=f, n=num_patches_per_frame)
                v_m = rearrange(v_m, '(b f) n c -> b f n c', f=f, n=num_patches_per_frame)

                k_m = torch.cat([k_m[:, [0] * f], k_m[:, ff_index]], dim=2)

                if type(self.sa_first_frame_scale) == float:
                    v_m = torch.cat([v_m[:, [0] * f] * self.sa_first_frame_scale, v_m[:, ff_index] * (1-self.sa_first_frame_scale)], dim=2)
                else:
                    if self.sa_first_frame_scale == "linear":
                        weight = (timestep.cpu().item()) / 1000
                    elif self.use_weight_module:
                        # get weighting from Adaptive Weight Module
                        weight = self.weight_module(t_emb)[0,0]
                    v_m = torch.cat([v_m[:, [0] * f] * weight, v_m[:, ff_index] * (1-weight)], dim=2)

                k_m = rearrange(k_m, 'b f n c -> (b f) n c', f=f)
                v_m = rearrange(v_m, 'b f n c -> (b f) n c', f=f)
                
        elif "prev" in self.sa_mode:

            x = rearrange(x, 'b (f n) c -> (b f) n c', f=f, n=num_patches_per_frame)
            k = self.norm_k(self.k(x))
            v = self.v(x)

            k = rearrange(k, '(b f) n c -> b f n c', f=f, n=num_patches_per_frame)
            v = rearrange(v, '(b f) n c -> b f n c', f=f, n=num_patches_per_frame)

            k = k[:, ff_index]
            v = v[:, ff_index]

            k = rearrange(k, 'b f n c -> (b f) n c', f=f, n=num_patches_per_frame)
            v = rearrange(v, 'b f n c -> (b f) n c', f=f, n=num_patches_per_frame)

            if self.use_mask_branch:
                x_m = rearrange(x_m, 'b (f n) c -> (b f) n c', f=f, n=num_patches_per_frame)
                k_m = self.norm_k(self.k(x_m))
                v_m = self.v(x_m)

                k_m = rearrange(k_m, '(b f) n c -> b f n c', f=f, n=num_patches_per_frame)
                v_m = rearrange(v_m, '(b f) n c -> b f n c', f=f, n=num_patches_per_frame)

                k_m = k_m[:, ff_index]
                v_m = v_m[:, ff_index]

                k_m = rearrange(k_m, 'b f n c -> (b f) n c', f=f, n=num_patches_per_frame)
                v_m = rearrange(v_m, 'b f n c -> (b f) n c', f=f, n=num_patches_per_frame)
    

        elif "first" in self.sa_mode:
            x = rearrange(x, 'b (f n) c -> b f n c', f=f, n=num_patches_per_frame)
            k = self.norm_k(self.k(x[:,0,:,:]))
            v = self.v(x[:,0,:,:])
            k = rearrange(k, 'b (f n) c -> (b f) n c', f=f)
            v = rearrange(v, 'b (f n) c -> (b f) n c', f=f)

            if self.use_mask_branch:
                x_m = rearrange(x_m, 'b (f n) c -> b f n c', f=f, n=num_patches_per_frame)
                k_m = self.norm_k(self.k(x_m[:,0,:,:]))
                v_m = self.v(x_m[:,0,:,:])
                k_m = rearrange(k_m, 'b (f n) c -> (b f) n c', f=f)
                v_m = rearrange(v_m, 'b (f n) c -> (b f) n c', f=f)
        
        # Q from all frames, KV from first frame
        q = rearrange(q, 'b (f n) c -> (b f) n c', f=f, n=num_patches_per_frame)
        if self.use_mask_branch:
            q_m = rearrange(q_m, 'b (f n) c -> (b f) n c', f=f, n=num_patches_per_frame)
        x = self.attn(q, k, v, attn_mask=attention_mask)
        if self.use_mask_branch:
            x_m = self.attn(q_m, k_m, v_m)
        x = rearrange(x, '(b f) n c -> b (f n) c', f=f, n=num_patches_per_frame)
        if self.use_mask_branch:
            x_m = rearrange(x_m, '(b f) n c -> b (f n) c', f=f, n=num_patches_per_frame)
        
        x = self.o(x)
        if self.use_mask_branch:
            x_m = self.o_m(x_m)
            x = torch.cat([x, x_m], dim=0)

        return x    

# CA with implicit prompt implementation
class Simple_CA(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, bias=False,
                num_adapters=1, rank=64, act="gelu", norm_mid="layernorm",
                use_out=False):
        super().__init__()  
        self.dim = dim

        self.q = nn.Linear(dim, dim, bias)
        self.norm_q = RMSNorm(dim, eps=eps)

        if act == "gelu":
            self.activation_fn = F.gelu
        else:
            self.activation_fn = nn.Identity

        if norm_mid == "layernorm":
            self.CA_norm = nn.LayerNorm(rank)
        else:
            self.CA_norm = nn.Identity

        if num_adapters > 1:
            self.CA2 = nn.ModuleList([nn.Linear(dim, rank, bias=bias) for _ in range(num_adapters)])
            self.CA3 = nn.ModuleList([nn.Linear(rank, dim, bias=bias) for _ in range(num_adapters)])
        else:
            self.CA2 = nn.Linear(dim, rank, bias=bias)
            self.CA3 = nn.Linear(rank, dim, bias=bias)

        self.use_out = use_out
        if use_out:
            self.CA4 = nn.Linear(dim, dim, bias)
            last_layer = self.CA4
        else:
            last_layer = self.CA3

        with torch.no_grad():
            for name, param in last_layer.named_parameters():
                torch.nn.init.zeros_(param.data)

    def forward(self,
                x,
                num_adapters=1,
                attention_mask=None):
        q = self.norm_q(self.q(x))
        
        if num_adapters > 1:
            res = []
            for i in range(self.num_adapters):
                cur = self.CA3[i](self.activation_fn(self.CA_norm(self.CA2[i](q))))
                if self.use_out:
                    cur = self.CA4(cur)
                res.append(cur * self.adapter_weights[i])
            x = sum(res)
        else:
            x = self.activation_fn(self.CA_norm(self.CA2(q)))
            x = self.CA3(x)
            if self.use_out:
                x = self.CA4(x)

        if attention_mask is not None:
            x = x * attention_mask
        return x