from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import cv2

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from ..models.utils import estimate_x0
from torchvision.transforms.functional import rgb_to_grayscale
import random
import torch.nn.functional as F

from .utils import *

# SA mask derivation
def interframe_mask(masks, f, sa_mode, device): # only support for batch_size equals one
    h, w = masks[0].shape
    hw = h * w
    first_frame_mask = torch.zeros((f, hw, hw)).to(device)
    for i in range(f):
        first_frame_mask[i] = generate_mask(masks[i], masks[0])
    
    prev_frame_mask = torch.zeros((f, hw, hw)).to(device)
    for i in range(1, f):
        prev_frame_mask[i] = generate_mask(masks[i], masks[i - 1])
    
    if "first" in sa_mode and "prev" in sa_mode:
        frame_mask = torch.cat([first_frame_mask, prev_frame_mask], dim=2)
    elif "first" in sa_mode:
        frame_mask = first_frame_mask
    else:
        frame_mask = prev_frame_mask

    # frame_mask[frame_mask == 0] = -10
    # frame_mask[frame_mask == 1] = 0
    eps = 1e-4
    frame_mask = torch.log(frame_mask + eps).to(device)
    frame_mask = torch.clamp(frame_mask, min=-10, max=0)

    return frame_mask

def generate_mask(m1, m2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    # return m1[:, None] == m2[None, :]
    return m1[:, None] * m2[None, :] + (1 - m1[:, None]) * (1 - m2[None, :])

# 3D full attention mask derivation
def get_SA_full_mask(masks, f, device): # only support for batch_size equals one
    h, w = masks[0].shape
    flat_mask = masks.flatten()
    attn_mask = flat_mask[:, None] * flat_mask[None, :] + (1 - flat_mask[:, None]) * (1 - flat_mask[None, :])
    attn_mask = attn_mask.to(device)

    # frame_mask[frame_mask == 0] = -10
    # frame_mask[frame_mask == 1] = 0
    eps = 1e-4
    attn_mask = torch.log(attn_mask + eps).to(device)
    attn_mask = torch.clamp(attn_mask, min=-10, max=0)

    return attn_mask.unsqueeze(0)

# CA attention mask derivation
def get_CA_mask(masks, f, device):
    h, w = masks[0].shape
    # attn_mask = torch.stack(masks.clone()).to(device)
    attn_mask = masks.clone().to(device)
    # attn_mask = attn_mask > 0.5

    return attn_mask

def temporal_compression(masks):
    f, a, b = masks.shape
    assert f == 17, f"Expected 17 frames, got {f}"
    
    # Keep the first frame
    first_frame = masks[0:1]  # shape (1, a, b)
    
    # Average the remaining 16 frames in groups of 4
    rest = masks[1:]         # shape (16, a, b)
    rest = rest.view(4, 4, a, b)      # shape (4 groups, 4 frames per group, a, b)
    averaged = rest.mean(dim=1)       # shape (4, a, b)

    # Concatenate first frame with compressed frames
    compressed = torch.cat([first_frame, averaged], dim=0)  # shape (5, a, b)
    return compressed

def quantile_binarization(masks):
    f, a, b = masks.shape
    ref_mask = masks[0]
    area_ratio = ref_mask.sum() / ref_mask.numel()

    for i in range(1, f):
        frame = masks[i].flatten()

        threshold = torch.quantile(frame, 1 - area_ratio)

        masks[i] = (masks[i] >= threshold).float()

    return masks

def otsu_binarization(masks):
    f, a, b = masks.shape
    masks = masks.detach().cpu().numpy()
    masks = (masks * 255).astype(np.uint16)

    for i in range(f):
        _, masks[i] = cv2.threshold(masks[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    masks = (masks > 0)

    return torch.from_numpy(masks).float()

def apply_guided_filter(masks):
    masks = masks.float().detach().cpu().numpy()
    f, a, b = masks.shape

    radius = 16
    eps = 1e-3

    for i in range(f):
        masks[i] = cv2.ximgproc.guidedFilter(masks[i], masks[i], radius, eps)
    
    return torch.from_numpy(masks)

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float32, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        # dtype = next(iter(self.text_encoder.parameters())).dtype
        # enable_vram_management(
        #     self.text_encoder,
        #     module_map = {
        #         torch.nn.Linear: AutoWrappedLinear,
        #         torch.nn.Embedding: AutoWrappedModule,
        #         T5RelativeEmbedding: AutoWrappedModule,
        #         T5LayerNorm: AutoWrappedModule,
        #     },
        #     module_config = dict(
        #         offload_dtype=dtype,
        #         offload_device="cpu",
        #         onload_dtype=dtype,
        #         onload_device="cpu",
        #         computation_dtype=self.torch_dtype,
        #         computation_device=self.device,
        #     ),
        # )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_mask=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=1.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,

        is_training=False,
        first_frame_cond_type="lamp",
        shared_noise_ratio=0,
        use_dct_init=False,
        dct_cutoff_ratio = 0.23,
        dct_cutoff_shape = 'rect',

        prompt_emb=None,
        use_mask_branch=False,
        mask_branch_kwargs={},
        sa_mode="first",
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        if use_mask_branch:
            noise_m = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
            noise_m = noise_m.to(dtype=self.torch_dtype, device=self.device)
            latents_m = noise_m

        # Encode prompts
        if prompt_emb is not None:
            print(f"[Pipeline]Loaded null-prompt.")
            prompt_emb_posi = prompt_emb_nega = prompt_emb
        else:
            print(f"[Pipeline]Encoding prompt = {prompt}")
            self.load_models_to_device(["text_encoder"])
            prompt_emb_posi = self.encode_prompt(prompt, positive=True)
            if cfg_scale != 1.0:
                prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        # if input_image is not None and self.image_encoder is not None:
        #     self.load_models_to_device(["image_encoder", "vae"])
        #     image_emb = self.encode_image(input_image, num_frames, height, width)
        # else:
        #     image_emb = {}

        # we are not using Wan image encoder, image_emb should be empty
        image_emb = {}
        
        if input_image is not None:
            self.load_models_to_device(["vae"])
            input_image = center_crop_image(input_image, width, height)
            input_image = self.preprocess_image(input_image.resize((width, height))).to(self.device)
            vae_input = torch.concat([input_image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(input_image.device)], dim=1)
            frame_emb = self.encode_video(vae_input.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device), **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

        if input_mask is not None:
            input_mask = center_crop_image(input_mask, width, height)
            input_mask = self.preprocess_image(input_mask.resize((width, height))).to(self.device)
            vae_input_m = torch.concat([input_mask.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(input_mask.device)], dim=1)
            mask_emb = self.encode_video(vae_input_m.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device), **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

        if is_training is not True:
            if first_frame_cond_type == "lamp":
                print("[Pipeline]Noise Init mode: lamp")
                first_frame_latents = frame_emb[:, :, 0:1, :, :]
                first_mask_latents = mask_emb[:, :, 0:1, :, :]
                # share noise ratio
                if shared_noise_ratio > 0:
                    print("[Pipeline]Shared noise ratio =", shared_noise_ratio)
                    for f in range(1, (num_frames - 1) // 4 + 1):
                        latents[:, :, f:f+1, :, :] = shared_noise_ratio * latents[:, :, 0:1, :, :] +\
                            (1-shared_noise_ratio) * latents[:, :, f:f+1, :, :]
                        if use_mask_branch:
                            latents_m[:, :, f:f+1, :, :] = shared_noise_ratio * latents_m[:, :, 0:1, :, :] +\
                                (1-shared_noise_ratio) * latents_m[:, :, f:f+1, :, :]
                latents[:, :, 0:1, :, :] = first_frame_latents
                if use_mask_branch:
                    latents_m[:, :, 0:1, :, :] = first_mask_latents
            else:
                first_frame_latents = frame_emb[:, :, 0:1, :, :]
                first_frame_latents = first_frame_latents.repeat(1, 1, (num_frames - 1) // 4, 1, 1)
                if use_mask_branch:
                    first_mask_latents = mask_emb[:, :, 0:1, :, :]
                    first_mask_latents = first_mask_latents.repeat(1, 1, (num_frames-1) // 4, 1, 1)
                if shared_noise_ratio > 0:
                    print("[Pipeline]Shared noise ratio =", shared_noise_ratio)
                    for f in range(1, (num_frames - 1) // 4 + 1):
                        latents[:, :, f:f+1, :, :] = shared_noise_ratio * latents[:, :, 0:1, :, :] +\
                            (1-shared_noise_ratio) * latents[:, :, f:f+1, :, :]
                        if use_mask_branch:
                            latents_m[:, :, f:f+1, :, :] = shared_noise_ratio * latents_m[:, :, 0:1, :, :] +\
                              (1-shared_noise_ratio) * latents_m[:, :, f:f+1, :, :]
                        
                diffuse_timesteps = torch.full((1,),int(975))
                diffuse_timesteps = diffuse_timesteps.long()
                noisy_base_content = self.scheduler.add_noise(first_frame_latents, latents[:,:,1:,:,:], diffuse_timesteps.to(self.device))
                if use_mask_branch:
                    noisy_base_content_m = self.scheduler.add_noise(first_mask_latents, latents_m[:,:,1:,:,:], diffuse_timesteps.to(self.device))

                # DCTInit:
                if use_dct_init:
                    print("[Pipeline]Noise Init mode: DCTInit")
                    freq_filter = dct_low_pass_filter(dct_coefficients=first_frame_latents,
                                                                percentage=dct_cutoff_ratio, cutoff_shape=dct_cutoff_shape)

                    latent_h, latent_w = latents.shape[-2], latents.shape[-1]
                    latents.resize_(latents.shape[0], latents.shape[1], latents.shape[2], 64, 64)
                    freq_filter.resize_(latents.shape[0], latents.shape[1], latents.shape[2]-1, 64, 64)
                    noisy_base_content.resize_(latents.shape[0], latents.shape[1], latents.shape[2]-1, 64, 64)

                    dct_latents = exchanged_mixed_dct_freq(noise=latents[:,:,1:,:,:],
                                base_content=noisy_base_content,
                                LPF_3d=freq_filter).to(dtype=torch.float32)

                    dct_latents.resize_(latents.shape[0], latents.shape[1], latents.shape[2]-1, latent_h, latent_w).to(dtype=self.torch_dtype, device=self.device)
                    if use_mask_branch:
                        freq_filter_m = dct_low_pass_filter(dct_coefficients=first_mask_latents,
                                                                percentage=dct_cutoff_ratio, cutoff_shape=dct_cutoff_shape)
                        latents_m.resize(latents.shape[0], latents.shape[1], latents.shape[2], 64, 64)
                        freq_filter_m.resize_(latents.shape[0], latents.shape[1], latents.shape[2]-1, 64, 64)
                        noisy_base_content_m.resize_(latents.shape[0], latents.shape[1], latents.shape[2]-1, 64, 64)

                        dct_latents_m = exchanged_mixed_dct_freq(noise=latents_m[:,:,1:,:,:],
                                    base_content=noisy_base_content_m,
                                    LPF_3d=freq_filter_m).to(dtype=torch.float32)

                else:
                    print("[Pipeline] Noise Init mode: Cinemo-no DCT")
                    dct_latents = noisy_base_content
                    if use_mask_branch:
                        dct_latents_m = noisy_base_content_m
                
                if use_mask_branch:
                    latents = torch.concat([first_frame_latents[:,:,0:1,:,:], dct_latents, first_mask_latents[:,:,0:1,:,:], dct_latents_m], dim=2)
                else:
                    latents = torch.concat([first_frame_latents[:,:,0:1,:,:], dct_latents],dim=2)
        
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(["dit"])
        last_masks = None
        use_mask_guidance = mask_branch_kwargs.get("use_mask_guidance", False)
        mask_guided_attention = mask_branch_kwargs.get("mask_guided_attention", "SA")
        binarization_mode = mask_branch_kwargs.get("binarization_mode", None)
        guided_filter = mask_branch_kwargs.get("guided_filter", False)
        video_length = (num_frames - 1) // 4 + 1
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            attention_masks = {}
            if use_mask_guidance and last_masks is not None:
                masks = last_masks.clone() 
                masks = temporal_compression(masks)

                if guided_filter:
                    masks = apply_guided_filter(masks)
                if binarization_mode == "otsu":
                    masks = otsu_binarization(masks)
                elif binarization_mode == "quantile":
                    masks = quantile_binarization(masks)
                
                if "SA" in mask_guided_attention:
                    attention_masks["SA"] = interframe_mask(masks, video_length, sa_mode, device="cuda").to(self.torch_dtype)
                if "CA" in mask_guided_attention:
                    attention_masks["CA"] = get_CA_mask(masks, video_length, device="cuda").to(self.torch_dtype) # (b, fhw) filter foreground
                if "full" in mask_guided_attention:
                    attention_masks["full"] = get_SA_full_mask(masks, video_length, device="cuda").to(self.pipe.torch_dtype)
            
            # Inference
            noise_pred_posi = model_fn_wan_video(self.dit, latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input, **tea_cache_posi, attention_masks=attention_masks)
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(self.dit, latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input, **tea_cache_nega, attention_masks=attention_masks)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents_new = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            latents[:, :, 1:, :, :] = latents_new[:, :, 1:, :, :]

            if use_mask_branch:
                estimate_masks = latents[:, :, video_length:, :, :].clone()
                for i in range(latents.shape[0]):
                    estimate_masks[i, :, 1:, :, :] = estimate_x0(latents[i,:,video_length+1:,:,:].squeeze(0), progress_id, self.scheduler, noise_pred[i,:,video_length+1:,:,:].squeeze(0))
                decoded_masks = self.decode_video(estimate_masks, **tiler_kwargs)
                b, c, f, height, width = decoded_masks.shape
                decoded_masks = decoded_masks.reshape(-1, 1, height, width)
                decoded_masks = F.interpolate(decoded_masks, size=(height//8, width//8), mode='area')
                decoded_masks = decoded_masks.reshape(c, f, height//8, width//8)
                estimate_masks = decoded_masks
                del decoded_masks
                estimate_masks = (estimate_masks.clamp(0, 1).squeeze().permute(1,2,3,0) * 255).to(torch.uint8)

                estimate_masks_bin = []
                for i, frame in enumerate(estimate_masks):
                    estimate_mask_bin = rgb_to_grayscale(frame.permute(2, 0, 1))
                    estimate_masks_bin.append(estimate_mask_bin / 255)
                estimate_masks_bin = torch.stack(estimate_masks_bin, dim=0)
                self.last_masks = estimate_masks_bin.squeeze(1)

        # Decode
        self.load_models_to_device(['vae'])
        if use_mask_branch:
            latents_m = latents[:, :, video_length:, :, :].clone()
            latents = latents[:, :, :video_length, :, :].clone()

        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        if use_mask_branch:
            masks = self.decode_video(latents_m, **tiler_kwargs)
            masks = masks[0]
            masks = rearrange(masks, "C T H W -> T C H W")
            masks = ((masks.float() + 1) * 127.5).clip(0, 255)
            masks = torch.stack([rgb_to_grayscale(mask) for mask in masks]).squeeze(1) / 255.0
            # print(f"shape: {masks.shape}")
            # raise SystemExit
            if guided_filter:
                masks = apply_guided_filter(masks)
            if binarization_mode == "otsu":
                masks = otsu_binarization(masks)
            elif binarization_mode == "quantile":
                masks = quantile_binarization(masks)
            
            masks = (masks * 255.0).cpu().numpy().astype(np.uint8)
            masks = [Image.fromarray(mask).convert("RGB") for mask in masks]
            
            frames = frames + masks

        return frames



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    attention_masks: dict = {},
    **kwargs,
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    if dit.use_mask_branch:
        x = rearrange(x, "b c (m f) h w -> (m b) c f h w", m=2)
    
    # x, (f, h, w) = dit.patchify(x)
    x, grid_size = dit.patchify(x)
    f, h, w = grid_size
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        # blocks
        for block in dit.blocks:
            x = block(x, context, t, t_mod, freqs, grid_size, timestep, attention_masks=attention_masks)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))

    if dit.use_mask_branch:
        x = rearrange(x, "(m b) c f h w -> b c (m f) h w", m=2)
    return x
