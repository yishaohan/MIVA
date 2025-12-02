import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attn_mask: torch.Tensor=None):
    # if compatibility_mode:
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    if attn_mask is not None:
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)

    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    # elif FLASH_ATTN_3_AVAILABLE:
    #     q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
    #     k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
    #     v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
    #     x = flash_attn_interface.flash_attn_func(q, k, v)
    #     x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    # elif FLASH_ATTN_2_AVAILABLE:
    #     q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
    #     k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
    #     v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
    #     x = flash_attn.flash_attn_func(q, k, v)
    #     x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    # elif SAGE_ATTN_AVAILABLE:
    #     q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    #     k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    #     v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    #     x = sageattn(q, k, v)
    #     x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    # else:
        # q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        # k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        # v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        # x = F.scaled_dot_product_attention(q, k, v)
        # x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attn_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, use_mask_branch = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.use_mask_branch = use_mask_branch

    def forward(self, x, freqs, attention_mask = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        if self.use_mask_branch:
            q = rearrange(q, "(m b) n c -> m b n c", m=2)
            k = rearrange(k, "(m b) n c -> m b n c", m=2)
            q = torch.concat([rope_apply(q[0], freqs, self.num_heads), rope_apply(q[1], freqs, self.num_heads)])
            k = torch.concat([rope_apply(k[0], freqs, self.num_heads), rope_apply(k[1], freqs, self.num_heads)])
        else:
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
    
        x = self.attn(q, k, v, attn_mask=attention_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)

# attention mask downscale to appropriate resolution
def SA_downsample(attn_mask, gridsize):
    _, height, width = gridsize
    attn_mask = attn_mask.unsqueeze(1)
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


def full_downsample(attn_mask, gridsize):
    f, h, w = gridsize
    b, fhw_old, _ = attn_mask.shape
    hw_old = fhw_old / f
    r_square = hw_old // (h * w)
    r = int(math.sqrt(r_square))
    h_old = h * r
    w_old = w * r

    # Step 1: Reshape to 6D tensor: [f, h, w, f, h, w]
    attn_mask = attn_mask.view(f, h_old, w_old, f, h_old, w_old)

    # Step 2: Merge batch dims for 2D interpolation on source axis
    attn_mask = attn_mask.permute(0, 3, 1, 2, 4, 5)  # [f, f, h, w, h, w]
    attn_mask = attn_mask.reshape(f * f, 1, h_old, w_old, h_old, w_old)

    # Downsample spatial dimensions (source side)
    attn_mask = attn_mask.reshape(-1, 1, h_old, w_old)
    attn_mask = F.interpolate(attn_mask, scale_factor=1/r, mode='area')
    attn_mask = attn_mask.reshape(f, f, h_old, w_old, h, w)

    # Downsample spatial dimensions (target side)
    attn_mask = attn_mask.permute(0, 1, 4, 5, 2, 3)  # [f, f, h, w, h', w']
    attn_mask = attn_mask.reshape(-1, 1, h_old, w_old)
    attn_mask = F.interpolate(attn_mask, scale_factor=1/r, mode='area')
    attn_mask = attn_mask.reshape(f, f, h, w, h, w)

    # Final reshape to 2D attention mask
    attn_mask = attn_mask.permute(0, 2, 3, 1, 4, 5)  # [f, h', w', f, h', w']
    attn_mask = attn_mask.reshape(f * h * w, f * h * w)

    attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)

    attn_mask_m = torch.zeros_like(attn_mask)
    attn_mask = torch.cat([attn_mask, attn_mask_m], dim=0)

    return attn_mask

def CA_downsample(attn_mask, gridsize):
    attn_mask = attn_mask.unsqueeze(1) #(f, 1, h, w)
    _, h, w = gridsize
    hw = h * w
    r_square = (attn_mask.shape[-2] * attn_mask.shape[-1]) // hw
    r = int(math.sqrt(r_square))
    attn_mask = F.interpolate(attn_mask, scale_factor=1/r, mode="area")
    attn_mask = rearrange(attn_mask, "f d h w -> (f h w) d")
    attn_mask = attn_mask.unsqueeze(0)

    attn_mask_m = torch.ones_like(attn_mask)
    attn_mask = torch.cat([attn_mask, attn_mask_m], dim=0)

    return attn_mask

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6,
                 SA_mode:str = "vanilla",
                 SA_kwargs:dict = {},
                 CA_mode:str = "vanilla",
                 CA_kwargs:dict = {},
                 use_mask_branch:bool = False,
                 mask_branch_kwargs:dict = {}):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps, use_mask_branch=use_mask_branch)

        self.SA_mode = SA_mode
        if SA_mode == "first":
            from .miva import I2VAdapter
            self.i2v_adapter = I2VAdapter(dim, num_heads, eps, use_mask_branch=use_mask_branch, mask_branch_kwargs=mask_branch_kwargs)
        else:
            from .miva import I2VAdapter
            self.i2v_adapter = I2VAdapter(dim, num_heads, eps, SA_mode, **SA_kwargs, use_mask_branch=use_mask_branch, mask_branch_kwargs=mask_branch_kwargs)

        self.CA_mode = CA_mode
        self.CA_mask_guidance = "CA" in mask_branch_kwargs.get("mask_guided_attention", "SA")
        if CA_mode == "vanilla": # vanilla
            self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        elif CA_mode == "simple":
            from .miva import Simple_CA
            self.cross_attn = Simple_CA(dim, num_heads, eps, **CA_kwargs)
            if self.CA_mask_guidance:
                self.cross_attn_b = Simple_CA(dim, num_heads, eps, **CA_kwargs)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.use_mask_branch = use_mask_branch

    def forward(self, x, context, t, t_mod, freqs, grid_size, timestep, attention_masks={}):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        full_mask = attention_masks.get("full", None)
        if full_mask is not None:
            full_mask = full_downsample(full_mask, grid_size)
        x = x + gate_msa * self.self_attn(input_x, freqs, attention_mask=full_mask)
        
        if self.SA_mode != "vanilla":
            # Apply i2v-adapter 
            SA_mask = attention_masks.get("SA", None)
            if SA_mask is not None:
                if "first" in self.SA_mode and "prev" in self.SA_mode:
                    SA_mask = rearrange(SA_mask, "b c (m d) -> (b m) c d", m=2)
                SA_mask = SA_downsample(SA_mask, grid_size)
                if "first" in self.SA_mode and "prev" in self.SA_mode:
                    SA_mask = rearrange(SA_mask, "(b m) h c d -> b h c (m d)", m=2)
                
            i2v_hidden_states = self.i2v_adapter(x, grid_size, timestep, t_emb=t, attention_mask=SA_mask)
            # i2v_hidden_states = torch.concat([self.i2v_adapter(x[0:1], grid_size, timestep, t_emb=t, attention_mask=SA_mask), self.i2v_adapter(x[1:2], grid_size, timestep, t_emb=t, attention_mask=SA_mask)])
            x = x + i2v_hidden_states

        if self.CA_mode == "vanilla":
            x = x + self.cross_attn(self.norm3(x), context)
        elif self.CA_mode == "simple": # simple_CA
            CA_mask = attention_masks.get("CA", None)
            if CA_mask is not None:
                CA_mask = CA_downsample(CA_mask, grid_size)
            x = x + self.cross_attn(self.norm3(x), attention_mask=CA_mask) + (self.cross_attn_b(self.norm3(x), attention_mask=1-CA_mask if CA_mask is not None else None) if self.CA_mask_guidance else 0)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int = 1536,
        in_dim: int = 16,
        ffn_dim: int = 8960,
        out_dim: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        
        patch_size: Tuple[int, int, int] = [1,2,2],
        num_heads: int = 12,
        num_layers: int = 30,
        has_image_input: bool = False,

        eps: float = 1e-6,

        SA_mode:str = "vanilla",
        SA_kwargs:dict = {},
        CA_mode:str = "vanilla",
        CA_kwargs:dict = {},
        use_mask_branch:bool = False,
        mask_branch_kwargs:dict = {},
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,
                     SA_mode, SA_kwargs,
                     CA_mode, CA_kwargs,
                     use_mask_branch, mask_branch_kwargs)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        self.use_mask_branch = use_mask_branch

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x) # b c f h w [1, 16, 5, 60, 104] -> [1, 1536, 5, 30, 52]
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous() # [1, 1536, 5, 30, 52] -> [1, 7800, 1536]
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                attention_masks:dict = {},
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        if self.use_mask_branch:
            x = rearrange(x, "b c (m f) h w -> (m b) c f h w", m=2)
        
        x, grid_size = self.patchify(x)
        f, h, w = grid_size
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t, t_mod, freqs, grid_size,
                            timestep,
                            attention_masks,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t, t_mod, freqs, grid_size,
                        timestep,
                        attention_masks,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t, t_mod, freqs, grid_size, attention_masks=attention_masks)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        
        if self.use_mask_branch:
            x = rearrange(x, "(m b) c f h w -> b c (m f) h w", m=2)
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config
