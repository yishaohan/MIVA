import torch
import os, sys
if os.path.abspath('.') not in sys.path:
    sys.path.insert(1, os.path.abspath('.'))
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
import argparse
from typing import Dict, Iterable
from omegaconf import OmegaConf
import glob
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)

EMPTY_TXT_EMB = torch.load("diffsynth/utilsempty_text_emb.pt")

def run_inference(
    output_dir: str,
    validation_data: Dict,
    validation: Dict,
    pretrained_model_path: str = "",
    adapter_path: str = "",
    num_persistent_param_in_dit: int = None,
    pipe: WanVideoPipeline = None,
    empty_txt_emb: bool = True,
    dit_kwargs: dict = {},
    use_mask_branch: bool = False,
    mask_branch_kwargs: dict = {},
    mask_only: bool = False
):

    # log 250409: currently we load models every validation step, which is not optimized
    if pipe is None:
        model_manager = ModelManager(device="cpu")
        model_manager.load_models(
            [
                # f"{pretrained_model_path}/diffusion_pytorch_model.safetensors",  # DiT Model
                # f"{pretrained_model_path}/models_t5_umt5-xxl-enc-bf16.pth",  # Text Encoder
                f"{pretrained_model_path}/Wan2.1_VAE.pth",  # VAE
            ],
            torch_dtype=torch.float32, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
        )

        dit_kwargs["use_mask_branch"] = use_mask_branch
        dit_kwargs["mask_branch_kwargs"] = mask_branch_kwargs
        # Loading DiT moved here
        dit_path = f"{pretrained_model_path}/diffusion_pytorch_model.safetensors"
        model_manager.load_dit(dit_path, adapter_path, dit_kwargs)
    
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.float32, device="cuda")
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)

    # Text-to-video
    # process the settings
    negative_prompt = validation_data.get("negative_prompt", None)
    num_frames = validation_data.get("video_length", 17)
    width = validation_data.get("width", 832)
    height = validation_data.get("height", 480)
    num_inference_steps =validation_data.get("num_inference_steps", 50)
    cfg_scale=validation_data.get("guidance_scale", 1.0)
    denoising_strength=validation_data.get("denoising_strength", 1.0)
    seed=validation_data.get("seed", None)
    rand_device=validation_data.get("rand_device", "cpu")
    sigma_shift=validation_data.get("sigma_shift", 5.0)
    tiled=validation_data.get("tiled", True)
    tile_size=validation_data.get("tile_size", (30, 52))
    tile_stride=validation_data.get("tile_stride", (15, 26))
    tea_cache_l1_thresh=validation_data.get("tea_cache_l1_thresh", None)
    tea_cache_model_id=validation_data.get("tea_cache_model_id", "")

    first_frame_cond_type=validation.get("first_frame_cond_type")
    shared_noise_ratio=validation.get("shared_noise_ratio")
    use_dct_init=validation.get("use_dct_init")
    dct_cutoff_ratio=validation.get("dct_cutoff_ratio")
    dct_cutoff_shape=validation.get("dct_cutoff_shape")

    os.makedirs(output_dir, exist_ok=True)

    # process the data path
    prompts = validation_data.prompt_path # the input images

    if prompts == "all":
        image_paths = glob.glob(f"{validation_data.image_path}/*.png")
    elif prompts == "newlamp":
        image_paths = [f"{validation_data.image_path}/{i}.png" for i in range(1,21)]
    elif isinstance(prompts, Iterable):
        image_paths = prompts.copy()
        for i, prompt in enumerate(prompts):
            img_name = prompt.replace(' ', '_')
            image_paths[i] = os.path.join(validation_data.image_path, img_name + '.png')
    else:
        raise NotImplementedError

    for i, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        img_name = os.path.basename(path)[:-4]

        if mask_only:
            image = Image.open(os.path.join(validation_data.mask_path, img_name + "_mask.jpg")).convert("RGB")

        if use_mask_branch:
            image_m = Image.open(os.path.join(validation_data.mask_path, img_name + "_mask.jpg")).convert("RGB")
        
        if hasattr(validation_data, "batch_prompt"):
            prompt = validation_data.batch_prompt
            if isinstance(prompt, Iterable):
                prompt = list(prompt)
                if len(prompt)==1:
                    prompt = prompt[0]
        elif isinstance(prompts, Iterable):
            prompt = validation_data.prompts[i]
        else:
            raise NotImplementedError
        
        prompt_emb = EMPTY_TXT_EMB if empty_txt_emb else None

        # Image-to-video
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=image,
            input_mask=image_m if use_mask_branch else None,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed, tiled=tiled,
            cfg_scale=cfg_scale,

            tea_cache_l1_thresh=tea_cache_l1_thresh, # The larger this value is, the faster the speed, but the worse the visual quality.
            tea_cache_model_id=tea_cache_model_id, # Choose one in (Wan2.1-T2V-1.3B, Wan2.1-T2V-14B, Wan2.1-I2V-14B-480P, Wan2.1-I2V-14B-720P).
            
            # inference trick
            first_frame_cond_type=first_frame_cond_type,
            shared_noise_ratio=shared_noise_ratio,
            use_dct_init=use_dct_init,
            dct_cutoff_ratio=dct_cutoff_ratio,
            dct_cutoff_shape=dct_cutoff_shape,

            prompt_emb=prompt_emb,
            use_mask_branch=use_mask_branch,
            mask_branch_kwargs=mask_branch_kwargs,
            sa_mode=dit_kwargs.get("SA_mode", "first")
        )
        target_path = os.path.join(output_dir, f"{img_name}.mp4")
        save_video(video, target_path, fps=8, quality=10)
        print("[inference.py]Video saved to:", target_path)

if __name__ == "__main__":
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    run_inference(**config)
