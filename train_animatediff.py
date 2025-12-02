import os
import math
import random
import logging
import inspect
import argparse

from tqdm.auto import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple
from accelerate import Accelerator

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.transforms.functional import rgb_to_grayscale
import safetensors

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from transformers import CLIPTextModel, CLIPTokenizer
from safetensors import safe_open
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline, interframe_mask, get_CA_mask, get_TA_mask, get_SA_original_mask
from animatediff.utils.util import save_videos_grid, zero_rank_print, save_progress
from lamp.data.dataset import LAMPDataset
from scripts.repeat_token import repeat_token
import cv2
import numpy as np
from PIL import Image
from animatediff.utils.loss import CalculateDiceLoss
from animatediff.models.seg_utils import gaussian_blur
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)

def set_trainable_modules(model, trainable_modules, keyword=None, output_path=None):
    model.requires_grad_(False)

    if output_path:
        lines = []
    
    # Set unet trainable parameters
    for name, module in model.named_modules():
        if name.endswith(tuple(trainable_modules)):
            if keyword is not None and keyword not in name:
                continue
            for params in module.parameters():
                params.requires_grad = True
                if output_path:
                    lines.append(name)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print("Trainable params written to:", output_path)

def save_tensor_as_pngs(tensor: torch.Tensor, output_dir="frames", mode="L"):
    """
    Saves each frame of a boolean tensor (f, h, w) as a PNG image.
    Each image will be black & white (0 or 255).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to uint8 format (0 or 255)
    tensor = tensor.to(torch.float16) * 255  # (f, h, w)
    tensor = tensor.to(torch.uint8)
    for i, frame in enumerate(tensor):
        img = Image.fromarray(frame.cpu().numpy(), mode=mode)  # 'L' for grayscale
        img.save(os.path.join(output_dir, f"frame_{i}.png"))

def main(
    image_finetune: bool,
    output_dir: str,
    pretrained_model_path: str,
    motion_module_path: str,
    use_ip_adapter: str,
    train_data: Dict,
    validation_data: Dict,
    ip_ckpt: str = None,
    image_encoder_path: str = None,
    repeat_token_mode: str = None,
    cfg_random_null_text: bool = False,
    cfg_random_null_text_ratio: float = 0.1,
    ckpt_mode: str = None, 
    
    unet_checkpoint_path: str = "",
    unet_checkpoint_starting_step: int = 0,
    unet_additional_kwargs: Dict = {},
    load_from_adapter_ckpt = False, 
    adapter_i2v_path: str = "",
    adapter_lora_path: str = "",
    ca_lora_path: str = "",
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,

    initializer_token: str = None,
    word_to_learn: str = None, # will learn the textual embedding of this word
    textemb_file_format: str = "safetensors", # if True, save the embedding as a "bin", else "safetensors"
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    mixed_precision: Optional[str] = "fp16",
    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    
    global_seed: int = 42,
    is_debug: bool = False,

    # training loss & BP behavior
    use_motion_distill_loss = False,
    use_8bit_adam: bool = True,
    include_frame1_in_mse_loss: bool = False,
    use_optical_flow = False,
    use_DDIMInv_first_frame_loss = False,
    use_DDIMInv_data = False,
    use_Mask_data = False,
    optical_flow_kwargs: Dict = {},
    use_perceptual_loss = False,
    perceptual_loss_kwargs: Dict = {},
    use_dino_loss = False,
    dino_loss_kwargs: Dict = {},
    log_losses_in_csv: bool = False, # by Mehdi
    print_trainable_params = False,
    use_dice_loss = False,
    use_DAVIS_data = False,

    pretrained_sa_adaptive_weight_module = None,
):
    check_min_version("0.10.0.dev0")

    seed = global_seed
    torch.manual_seed(seed)
    
    # Logging folder
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # if accelerator.is_main_process and (not is_debug):
    #     run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    do_clip_image_emb = unet_additional_kwargs.get("motion_module_kwargs", {}).get("simple_CA_kwargs", {}).get("use_clip_image_emb", False)

    if do_clip_image_emb:
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
            "cuda", dtype=torch.float16
        )
        image_processor = CLIPImageProcessor()
        print("[train] CLIP image encoder loaded.")

    unet_additional_kwargs["use_ip_adapter"] = use_ip_adapter
    # load resolution from "train_data" section
    width = train_data["width"]
    height = train_data["height"]
    unet_additional_kwargs["video_resolution"] = (width, height)

    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            motion_module_path,
             subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")


    if use_optical_flow:
        from animatediff.utils.loss import flow_to_image, Warping_Loss
        Warping_loss_instance = Warping_Loss(optical_flow_kwargs.raft_path)
        noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)  # num_inference_steps = train steps  , we need scheduler to get x_{t-1} when ReFL is not used

    if use_perceptual_loss:
        from animatediff.utils.loss import VGGLoss
        perceptual_loss_instance = VGGLoss(perceptual_loss_kwargs.vgg_path)
    
    if use_dino_loss:
        from animatediff.utils.loss import DINOLoss
        dino_loss_instance = DINOLoss(dino_loss_kwargs.dino_path)

    # Copy the weights form the attn1 SA layer to the i2v SA layer
    naive_sa_mode = unet_additional_kwargs.get('sa_mode', 'first') == 'first_naive'
    if naive_sa_mode:
        print("[train.py]NAIVE MODE ON.")
    with torch.no_grad():
        for name, param in unet.named_parameters():
            if 'i2v_adapter' in name and 'lora' not in name and "weight_module" not in name:
                source_w = name.replace("to_q_mask", "to_q")
                source_w = source_w.replace("to_out_mask", "to_out")
                source_w = source_w.replace("processor.", "")
                source_w = source_w.replace('i2v_adapter', 'attn1')
                param.copy_(unet.state_dict()[source_w])
            
            if 'CA1' in name:
                source_w = name.replace("sCA.CA1", "attn2.to_q")
                source_w = source_w.replace("sCA_b.CA1", "attn2.to_q")
                param.copy_(unet.state_dict()[source_w])

            if 'CA4' in name:
                source_w = name.replace("sCA.CA4", "attn2.to_out.0")
                source_w = source_w.replace("sCA_b.CA4", "attn2.to_out.0")
                param.copy_(unet.state_dict()[source_w])    

            if ('i2v_adapter.to_out' in name or 'attn_tCA.to_out' in name or 'CA3' in name) and not naive_sa_mode:
                torch.nn.init.zeros_(param.data)

    # Load pretrained unet weights
    if unet_checkpoint_path != "" and isinstance(unet_checkpoint_path, str):
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        state_dict = {}
        if ckpt_mode == "adapter":
            state_dict = torch.load(unet_checkpoint_path)
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        else:
            with safe_open(unet_checkpoint_path,framework="pt",device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if load_from_adapter_ckpt:
        state_dict = {}
        zero_rank_print(f"from split checkpoint: {adapter_i2v_path} and {adapter_lora_path}") 
        adapter_i2v_path = torch.load(adapter_i2v_path, map_location="cpu") 
        if "global_step" in adapter_i2v_path: zero_rank_print(f"global_step: {adapter_i2v_path['global_step']}")
        state_dict_i2v = adapter_i2v_path["state_dict"] if "state_dict" in adapter_i2v_path else adapter_i2v_path

        adapter_lora_path = torch.load(adapter_lora_path, map_location="cpu") 
        if "global_step" in adapter_lora_path: zero_rank_print(f"global_step: {adapter_lora_path['global_step']}")
        state_dict_lora = adapter_lora_path["state_dict"] if "state_dict" in adapter_lora_path else adapter_lora_path

        ca_lora_path = torch.load(ca_lora_path, map_location="cpu") 
        if "global_step" in ca_lora_path: zero_rank_print(f"global_step: {ca_lora_path['global_step']}")
        state_dict_ca_lora = ca_lora_path["state_dict"] if "state_dict" in ca_lora_path else ca_lora_path

        for k in state_dict_lora.keys():
            state_dict[k] = state_dict_lora[k]
        for k in state_dict_i2v.keys():
            state_dict[k] = state_dict_i2v[k]
        for k in state_dict_ca_lora.keys():
            state_dict[k] = state_dict_ca_lora[k]
        

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        assert len(u) == 0

    # Check and assign all their datat types
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # use IP adapter
    if use_ip_adapter == "IP-Adapter":
        from ip_adapter import IPAdapter
        ip_adapter = IPAdapter(unet, image_encoder_path, ip_ckpt, torch.device("cuda"))
        image_proj_model = ip_adapter.image_proj_model

        ip_state_dict = torch.load(ip_ckpt, map_location="cpu")
        image_proj_model.load_state_dict(ip_state_dict["image_proj"])

        ip_layers = {**unet.find_layers(keyword="to_k_ip"), **unet.find_layers(keyword="to_v_ip")}

        # sort key to order layers: down -> up -> mid
        def layer_sort_key(layer_name):
            tokens = layer_name.split('.')
            block_type = tokens[0]
            if block_type == 'down_blocks':
                return (0, int(tokens[1]))
            elif block_type == 'up_blocks':
                return (1, int(tokens[1]))
            elif block_type == 'mid_block':
                return (2, 0)
            else:
                return (3, 0) 
            
        # ablation study - load frozen SA adaptive weight module from an existing ckpt
        if pretrained_sa_adaptive_weight_module is not None:
            from scripts.load_sa_adaptive_weight_module import load_sa_adaptive_weight_module
            load_sa_adaptive_weight_module(unet, pretrained_sa_adaptive_weight_module)

        sorted_ip_layers = sorted(ip_layers.items(), key=lambda item: layer_sort_key(item[0]))
        
        sorted_weight_keys = sorted(ip_state_dict['ip_adapter'].keys(), key=lambda key: int(key.split('.')[0]))

        # load weights
        for (layer_name, layer_module), weight_key in zip(sorted_ip_layers, sorted_weight_keys):
            weight_tensor = ip_state_dict['ip_adapter'][weight_key]
            if layer_module.weight.shape == weight_tensor.shape:
                with torch.no_grad():
                    layer_module.weight.copy_(weight_tensor)

        print("IP adapter initialized")


    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # add textual inversion part here, specify word_to_learn in config file:
    if word_to_learn is not None:
        if unet_checkpoint_path == "":
            num_added_tokens = tokenizer.add_tokens([word_to_learn])
            initializer_token_id = tokenizer.encode(initializer_token, add_special_tokens=False)[0]

            added_token_ids = tokenizer.convert_tokens_to_ids([word_to_learn])
            text_encoder.resize_token_embeddings(len(tokenizer))

            token_embeds = text_encoder.get_input_embeddings().weight.data
            with torch.no_grad():
                for token_id in added_token_ids:
                    token_embeds[token_id] = token_embeds[initializer_token_id].clone()
        else:
            # loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
            TI_checkpoint_path = os.path.dirname(os.path.dirname(unet_checkpoint_path)) + "/" + "learned_embeds.safetensors"
            loaded_learned_embeds = safetensors.safe_open(TI_checkpoint_path, framework="pt", device='cpu')

            # separate token and the embeds
            trained_token = list(loaded_learned_embeds.keys())[0]
            embeds = loaded_learned_embeds.get_tensor(trained_token)

            # cast to dtype of text_encoder
            dtype = text_encoder.get_input_embeddings().weight.dtype
            embeds.to(dtype)

            # add the token in tokenizer
            token = trained_token
            num_added_tokens = tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
            
            # resize the token embeddings
            text_encoder.resize_token_embeddings(len(tokenizer))
            
            # get the id for the token and assign the embeds
            added_token_ids = tokenizer.convert_tokens_to_ids([token])
            for token_id in added_token_ids:
                text_encoder.get_input_embeddings().weight.data[token_id] = embeds
            print("Loaded pre-trained TI embedding.")

        # Freeze all parameters except for the token embeddings in text encoder
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)

    # Set unet trainable parameters
    path_trainable_params = os.path.join(output_dir, 'trainable_params.txt') if print_trainable_params else None
    set_trainable_modules(unet, trainable_modules, output_path = path_trainable_params)
    
    # OA training (deprecated)
    # log 240823: In OA case, freeze B here.
    if hasattr(unet_additional_kwargs, 'oa_bin_id'):
        from animatediff.models.lora_utils import replace_oa
        bin_id_from_config = unet_additional_kwargs.oa_bin_id
        dir_bases = unet_additional_kwargs.get("oa_bases_path", "AnimateDiff/configs/lora_bases_withca")
        lora_dict = unet.lora_layers()
        replace_oa(lora_dict, dir_bases, bin_id=bin_id_from_config, ignore_ca_kv=True)
        
    # Check trainiable params
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if word_to_learn is not None:
        optimizer = optimizer_cls(
            [{'params': unet.parameters()}, 
            {'params': text_encoder.get_input_embeddings().parameters()}
            ],
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # if accelerator.is_main_process:
    #     zero_rank_print(f"trainable params number: {len(trainable_params)}")
    #     zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    # if enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         unet.disable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")
    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if word_to_learn is not None:
            text_encoder.gradient_checkpointing_enable()

    # Move models to GPU
    if word_to_learn is None:
        text_encoder.to(accelerator.device,dtype=weight_dtype)
    vae.to(accelerator.device,dtype=weight_dtype)

    # Configure the ip adapter

    # Get the training dataset
    if use_DDIMInv_data:
        from animatediff.data.dataset import DDIMInv
        train_dataset = DDIMInv(**train_data)
    elif use_Mask_data:
        from animatediff.data.dataset import MaskDataset
        train_dataset = MaskDataset(**train_data)
    elif use_DAVIS_data:
        from lamp.data.dataset import DAVISDataset
        train_dataset = DAVISDataset(**train_data)
    else:
        train_dataset = LAMPDataset(**train_data)
    

    # Preprocessing the dataset
    for p in train_dataset.prompt:
        text_input_ids = tokenizer(
            p,
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids

        if repeat_token_mode is not None:
            text_embeddings = repeat_token(repeat_token_mode, text_input_ids, tokenizer, text_encoder)
        else:
            text_embeddings = text_encoder(text_input_ids.to('cuda'))[0]

        train_dataset.prompt_ids.append(text_embeddings[0])
    
    # del tokenizer
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # Prepare everything with our `accelerator`.
    if word_to_learn is not None:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil((max_train_steps - unet_checkpoint_starting_step) / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    if accelerator.is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num Trainable Parameters = {trainable_params / 1e6} M")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = unet_checkpoint_starting_step
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    if word_to_learn is not None:
        orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    if log_losses_in_csv:
        import pandas as pd
        loss_logger = pd.DataFrame(columns=['iter', 'time_step', 'loss_type', 'value'])
    else:
        loss_logger = None

    use_mask_branch = unet_additional_kwargs.get("use_mask_branch", False)
    for epoch in range(first_epoch, num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        if word_to_learn is not None:
            text_encoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['prompt_ids'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['prompt_ids']]
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['prompt_ids']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'animatediff-lamp-batch0' if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###
            
            def train_step(train_loss, loss_logger=None): # log 240815: ablate the codes for training FP+BP
                nonlocal global_step
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                if use_ip_adapter == "IP-Adapter":
                    first_frame = ((pixel_values[:,0, :, :] + 1.0) * 255/2).to(torch.uint8)

                if do_clip_image_emb:
                    key_idx = random.randint(0, 15)
                    frame = pixel_values.squeeze(0)[key_idx, :, :, :]
                    frame = image_processor(images=(frame + 1.0) / 2, return_tensors="pt", do_rescale=False)["pixel_values"].to("cuda")
                    clip_image_embeds = image_encoder(frame).image_embeds
                else:
                    clip_image_embeds = None

                # Convert videos to latent space            
                # this runs the vae on the BFCHW tensor
                with torch.no_grad():
                    if use_mask_branch:
                        masks_value = batch["masks_rgb"].to(weight_dtype)
                        pixel_values = torch.cat([pixel_values, masks_value], dim = 1)

                    if not image_finetune:
                        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                        latents = vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length * (2 if use_mask_branch else 1))
                    else:
                        latents = vae.encode(pixel_values).latent_dist
                        latents = latents.sample()

                    latents = latents * 0.18215
                
                bsz = latents.shape[0]
                
                # Get the text embedding for conditioning
                # log 250304: batch['prompt_ids'] is actually a text embedding instead of input ids like before, required by repeat token implementation
                if repeat_token_mode == "noise_after":
                    m = 76 # hard coded number of padding tokens
                    beta_txt = 0.02
                    noise_scale = beta_txt ** 0.5
                    text_noise = (torch.randn((train_batch_size, m, 768)) * noise_scale)
                    text_noise = torch.cat([torch.zeros(train_batch_size, 1, 768, device=text_noise.device, dtype=text_noise.dtype), text_noise], dim=1)
                    encoder_hidden_states = batch['prompt_ids'] + text_noise.to('cuda')
                else:                    
                    encoder_hidden_states = batch['prompt_ids']

                # Get the first frame embedding for IP adapter conditioning
                if use_ip_adapter == "IP-Adapter":
                    clip_image = ip_adapter.clip_image_processor(images = first_frame,return_tensors = "pt").pixel_values
                    clip_image_embeds = ip_adapter.image_encoder(clip_image.to(torch.device("cuda"),dtype=torch.float16)).image_embeds
                    image_prompt_embeds = ip_adapter.image_proj_model(clip_image_embeds)        

                else:
                    image_prompt_embeds = None

                mask_branch_kwargs = unet_additional_kwargs.get("mask_branch_kwargs", dict())
                num_iterations_per_data = mask_branch_kwargs.get("num_iterations_per_data", 1)
                ground_truth_mask_probability = mask_branch_kwargs.get("ground_truth_mask_probability", 1)
                use_mask_guidance = mask_branch_kwargs.get("use_mask_guidance", False)
                blurring = mask_branch_kwargs.get("blurring", False)
                mask_guided_attention = mask_branch_kwargs.get("mask_guided_attention", "SA")
                timesteps = torch.stack([
                                torch.randperm(noise_scheduler.config.num_train_timesteps, device=latents.device)[:num_iterations_per_data].sort(descending=True).values
                                for _ in range(bsz)
                            ], dim=0)

                if ground_truth_mask_probability == "cosine":
                    ground_truth_mask_probability = 0.5 * (1 + cos( (global_step / max_train_steps) * pi))

                # # Sample a random timestep for each video
                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                # # timesteps = torch.tensor(noise_scheduler.config.num_train_timesteps - 1, device=latents.device)
                # timesteps = timesteps.long()
                last_masks = None
                for i in range(num_iterations_per_data):
                    ts = timesteps[:, i]
                    ts = ts.long()
                    # Sample noise that we'll add to the latents
                    if use_DDIMInv_data:
                        inverted_latents = batch["latent_values"]
                        inverted_latents = rearrange(inverted_latents, "b f c h w -> b c f h w")
                        alpha_T = noise_scheduler.alphas_cumprod[-1]
                        noise = (inverted_latents - torch.sqrt(alpha_T) * latents) / torch.sqrt(1 - alpha_T)
                    else:
                        noise = torch.randn_like(latents)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, ts)

                    #replace the first frame of the noisy_latent with the unoised version
                    if not use_DDIMInv_data: # leave as option later (Hen 03/17/2025)
                        noisy_latents[:,:,0:1,:,:] = latents[:,:,0:1,:,:]
                        if use_mask_branch:
                            noisy_latents[:, :, video_length:video_length+1, :, :] = latents[:, :, video_length:video_length+1, :, :]
                    # Get the text embedding for conditioning

                    # Get the target for loss depending on the prediction type
                    # Animate diff only supports epsilon model noise parametrization
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        raise NotImplementedError
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if use_ip_adapter == "IP-Adapter":
                        encoder_hidden_states = torch.concat((image_prompt_embeds,  encoder_hidden_states), dim=1)

                    attention_masks = {}
                    if use_mask_guidance and i != 0 random.random() < ground_truth_mask_probability:
                        if random.random() < ground_truth_mask_probability or last_masks is None:
                            masks = batch["masks_bin"].squeeze(0)
                        else:
                            masks = last_masks
                        if blurring:
                            masks = gaussian_blur(masks.float(), kernel_size=11, sigma=1.0).to(weight_dtype)
                        if "SA" in mask_guided_attention:
                            attention_masks["SA"] = interframe_mask(masks, video_length, unet_additional_kwargs.get("sa_mode", "first"), device="cuda")
                        if "CA" in mask_guided_attention:
                            attention_masks["CA"] = get_CA_mask(masks, video_length, device="cuda") # (bf, hw) filter foreground
                        if "TA" in mask_guided_attention:
                            attention_masks["TA"] = get_TA_mask(masks, video_length, device="cuda") # (bhw, f, f)
                        if "original" in mask_guided_attention:
                            attention_masks["SA_original"] = get_SA_original_mask(masks, video_length, device="cuda")

                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, ts, encoder_hidden_states, attention_masks=attention_masks if (use_mask_guidance and i != 0) else {}, image_embeds=clip_image_embeds if do_clip_image_emb else None).sample                    
                    # log 24.07: motion distillation loss from [VMC Loss]
                    if use_motion_distill_loss:
                        delta_noise_pred = torch.abs(model_pred[:,:,1:,:,:] - model_pred[:,:,:-1,:,:])
                        delta_noise = torch.abs(noise[:,:,1:,:,:] - noise[:,:,:-1,:,:])
                        distill_loss = 0
                        distill_loss = 1 - (F.cosine_similarity(delta_noise_pred,delta_noise,dim=2)).mean()
                        loss = distill_loss
                    else:
                        if include_frame1_in_mse_loss:
                            loss = F.mse_loss(model_pred[:,:,:,:,:].float(), target[:,:,:,:,:].float(), reduction="mean")
                        else:
                            loss = F.mse_loss(model_pred[:,:,1:,:,:].float(), target[:,:,1:,:,:].float(), reduction="mean")

                    # print(f"pred: {model_pred[0, 0, 1, :10, :10]}, target: {target[0, 0, 1, :10, :10]}")

                    if log_losses_in_csv:
                        new_row = {"iter": global_step, "time_step": ts[0].cpu().detach().numpy(), 'loss_type': 'MSE', 'value': loss.item()}
                        loss_logger = pd.concat([loss_logger, pd.DataFrame([new_row])], ignore_index=True)

                    # cosine factor (can be used by some loss instances)
                    cosine_loss_factor = torch.square(torch.cos(1.5708 * ts / noise_scheduler.config.num_train_timesteps))
    
                    if use_perceptual_loss:
                        perceptual_loss_weight = perceptual_loss_kwargs.get('weight', 1)
                        perceptual_loss_cutoff = perceptual_loss_kwargs.get('cutoff', 1e6)

                        noisy_video_x0, target, perceptual_loss = perceptual_loss_instance.compute_perceptual_loss(noisy_latents, pixel_values, ts, noise_scheduler, model_pred, use_refl=True, vae=vae, **perceptual_loss_kwargs)                
                        c = cosine_loss_factor if perceptual_loss_kwargs.get('use_cosine_schedule', False) else 1
                        if perceptual_loss_kwargs.get('visualize', False):
                            import matplotlib.pyplot as plt
                            if global_step % 200 == 0:
                                select_idx = random.randint(0, 14)
                                img1 = noisy_video_x0[select_idx].permute(1, 2, 0).detach().cpu().numpy()  # Convert to NumPy
                                img2 = target[select_idx].permute(1, 2, 0).detach().cpu().numpy()  # Convert to NumPy
                                total_img = np.concatenate([img1, img2], axis=1)
                                plt.imshow(total_img)
                                plt.title(f'Time Step: {ts[0].cpu().detach().numpy()}, index: {select_idx}, prediction, GT')
                                plt.savefig(f'{output_dir}/perceptual_{global_step}_{select_idx}_viz.png')
                                plt.close()
                        if ts <= perceptual_loss_cutoff:
                            loss = torch.add(loss, perceptual_loss * perceptual_loss_weight * c)

                        if log_losses_in_csv:
                            new_row = {"iter": global_step, "time_step": ts[0].cpu().detach().numpy(), 'loss_type': 'Perceptual', 'value': perceptual_loss.item()}
                            loss_logger = pd.concat([loss_logger, pd.DataFrame([new_row])], ignore_index=True)

                    if use_dino_loss:
                        dino_loss_weight = dino_loss_kwargs.get('weight', 1)
                        noisy_video_x0, target, dino_loss = dino_loss_instance.compute_dino_loss(noisy_latents, pixel_values, ts, noise_scheduler, model_pred, use_refl=True, vae=vae, **dino_loss_kwargs)                
                        c = cosine_loss_factor if dino_loss_kwargs.get('use_cosine_schedule', False) else 1
                        if dino_loss_kwargs.get('visualize', False):
                            import matplotlib.pyplot as plt
                            if global_step % 200 == 0:
                                select_idx = random.randint(0, 14)
                                img1 = noisy_video_x0[select_idx].permute(1, 2, 0).detach().cpu().numpy()  # Convert to NumPy
                                img2 = target[select_idx].permute(1, 2, 0).detach().cpu().numpy()  # Convert to NumPy
                                total_img = np.concatenate([img1, img2], axis=1)
                                plt.imshow(total_img)
                                plt.title(f'Time Step: {ts[0].cpu().detach().numpy()}, index: {select_idx}, prediction, GT')
                                plt.savefig(f'{output_dir}/dino_{global_step}_{select_idx}_viz.png')
                                plt.close()   
                        
                        loss = torch.add(loss, dino_loss * dino_loss_weight)
                        if log_losses_in_csv:
                            new_row = {"iter": global_step, "time_step": ts[0].cpu().detach().numpy(), 'loss_type': 'DINO', 'value': dino_loss.item()}
                            loss_logger = pd.concat([loss_logger, pd.DataFrame([new_row])], ignore_index=True)
                            

                    if use_optical_flow:
                        warping_loss_weight = optical_flow_kwargs.get('weight', 1)
                        frames1, frames2, flows, warped_frames, boundary_mask, warping_loss = Warping_loss_instance.compute_warping_loss(noisy_latents, ts, noise_scheduler, model_pred, vae, 
                                                                                                                                        **optical_flow_kwargs)
                        c = cosine_loss_factor if optical_flow_kwargs.get('use_cosine_schedule', False) else 1

                        # TV loss of OF
                        if optical_flow_kwargs.get('use_tv_loss', False):
                            tv_weight = optical_flow_kwargs.get('weight_tv', 0.05)
                            tv_c = cosine_loss_factor if optical_flow_kwargs.get('use_cosine_schedule_tv', False) else 1
                            tv_loss_on_optical_flow = Warping_loss_instance.compute_total_variation_loss(noisy_latents, ts, noise_scheduler, model_pred, vae, **optical_flow_kwargs)
                            loss = torch.add(loss, tv_loss_on_optical_flow * tv_weight * tv_c)
                            if log_losses_in_csv:
                                new_row = {"iter": global_step, "time_step": ts[0].cpu().detach().numpy(), 'loss_type': 'TV', 'value': tv_loss_on_optical_flow.item()}
                                loss_logger = pd.concat([loss_logger, pd.DataFrame([new_row])], ignore_index=True)
                                
                        loss = torch.add(loss, warping_loss * warping_loss_weight * c)
                        if log_losses_in_csv:
                            new_row = {"iter": global_step, "time_step": ts[0].cpu().detach().numpy(), 'loss_type': 'Warping', 'value': warping_loss.item()}
                            loss_logger = pd.concat([loss_logger, pd.DataFrame([new_row])], ignore_index=True)
                            
                    
                        # visualizing samples
                        if optical_flow_kwargs.get('visualize', False):
                            import matplotlib.pyplot as plt
                            if global_step % 200 == 0:
                                select_idx = random.randint(0, 14)
                                # # Convert first optical flow frame to image
                                flow_img = flow_to_image(flows[select_idx].detach().cpu().numpy())
                                img1 = frames1[select_idx].permute(1, 2, 0).detach().cpu().numpy()  # Convert to NumPy
                                img1 = (img1 + 1) / 2.
                                img2 = frames2[select_idx].permute(1, 2, 0).detach().cpu().numpy()  # Convert to NumPy
                                img2 = (img2 + 1) / 2.
                                img3 = warped_frames[select_idx].permute(1, 2, 0).detach().cpu().numpy()  # Convert to NumPy
                                img3 = (img3 + 1) / 2.
                                bmask = torch.stack([boundary_mask[0, 0, :, :], boundary_mask[0, 0, :, :], boundary_mask[0, 0, :, :]], axis=-1)
                                bmask = (bmask.detach().cpu().numpy())
                                total_img = np.concatenate([img2, img1, img3, flow_img[:, :, ::-1], bmask], axis=1)

                                plt.imshow(total_img)
                                plt.title(f'Time Step: {ts[0].cpu().detach().numpy()}, index: {select_idx}, frame2, frame1, warped_frame, OF, mask')
                                plt.savefig(f'{output_dir}/{global_step}_{select_idx}_viz.png')
                                
                            plt.imshow(total_img)
                            plt.title(f'Time Step: {ts[0].cpu().detach().numpy()}, index: {select_idx}, frame2, frame1, warped_frame, OF, mask')
                            plt.savefig(f'{output_dir}/{global_step}_{select_idx}_viz.png')

                    if use_DDIMInv_first_frame_loss: # for inversion case - force the model to reconstruct 1st frame
                        from animatediff.utils.loss import estimate_x0
                        predict_frame = torch.stack([estimate_x0(noisy_latents[i,:,0:1,:,:].squeeze(0), ts, noise_scheduler, model_pred[i,:,0:1,:,:].squeeze(0)) for i in range(noisy_latents.shape[0])])
                        reconstruction_loss = F.mse_loss(latents[:,:,0:1,:,:].float(), predict_frame)
                        loss = torch.add(loss, reconstruction_loss)

                    if use_mask_branch:
                        with torch.no_grad():
                            from animatediff.utils.loss import estimate_x0
                            # DICE Loss and ReFL for mask generation
                            estimate_masks = noisy_latents[:, :, video_length:, :, :].clone()
                            for i in range(noisy_latents.shape[0]):
                                estimate_masks[i, :, 1:, :, :] = estimate_x0(noisy_latents[i,:,video_length+1:,:,:].squeeze(0), ts, noise_scheduler, model_pred[i,:,video_length+1:,:,:].squeeze(0))
                            decoded_masks = []
                            for i in range(0, video_length):
                                decode_mask = validation_pipeline.decode_latents(estimate_masks[:, :, i:i+1, :, :], refl=True, normalize=True)
                                decode_mask = decode_mask.squeeze(2)
                                _, _, height, width = decode_mask.shape
                                decode_mask = F.interpolate(decode_mask, size=(height // 8, width // 8), mode='area')
                                decode_mask = decode_mask.permute(1, 0, 2, 3)
                                decoded_masks.append(decode_mask)
                                torch.cuda.empty_cache()
                            estimate_masks = torch.stack(decoded_masks, dim=1)
                            del decoded_masks
                            estimate_masks = (estimate_masks.clamp(0, 1).squeeze().permute(1,2,3,0) * 255).to(torch.uint8)
                        # os.makedirs(f"./temp_estimate_masks", exist_ok=True)
                        # for i, frame in enumerate(estimate_masks):
                        #     img = Image.fromarray(frame.cpu().numpy())
                        #     img.save(f"./temp_estimate_masks/frame_new_{i}.png")

                        estimate_masks_bin = []
                        for i, frame in enumerate(estimate_masks):
                            estimate_mask_bin = rgb_to_grayscale(frame.permute(2, 0, 1))
                            estimate_masks_bin.append(estimate_mask_bin / 255)
                        estimate_masks_bin = torch.stack(estimate_masks_bin, dim=0)
                        save_tensor_as_pngs(estimate_masks_bin.squeeze(1) > 0.5, "./temp_estimate_masks")
                        # save_tensor_as_pngs(batch["masks_bin"].squeeze(0), "./temp_actual_masks")
                        # save_tensor_as_pngs(rearrange(batch["masks_rgb"].squeeze(0), "f c h w -> f h w c"), "./temp_actual_masks_rgb", mode="RGB")
                        # print(CalculateDiceLoss(estimate_masks_bin > 0.5, batch["masks_bin"].squeeze(0).unsqueeze(1)))

                        key_idx = random.randint(1, video_length - 1)
                        estimate_key_mask = estimate_x0(noisy_latents[0,:,video_length+key_idx:video_length+key_idx+1,:,:].squeeze(0), ts, noise_scheduler, model_pred[0,:,video_length+key_idx:video_length+key_idx+1,:,:].squeeze(0))
                        decode_mask = validation_pipeline.decode_latents(estimate_key_mask.unsqueeze(0).to(weight_dtype), refl=True, normalize=True)
                        decode_mask = decode_mask.squeeze(2)
                        _, _, height, width = decode_mask.shape # 1 3 40 64
                        decode_mask = F.interpolate(decode_mask, size=(height // 8, width // 8), mode='area')
                        decode_mask = decode_mask.permute(1, 0, 2, 3) # 3 1 40 64
                        decode_mask = (decode_mask.clamp(0, 1).permute(1,2,3,0) * 255).to(torch.uint8) # 1 40 64 4
                        decode_mask = rgb_to_grayscale(decode_mask.squeeze(0).permute(2, 0, 1)).unsqueeze(0) # 1 40 64
                        decode_mask = decode_mask / 255
                        if use_dice_loss and use_mask_guidance:
                            dice_loss = CalculateDiceLoss(decode_mask > 0.5, batch["masks_bin"].squeeze(0).unsqueeze(1)[key_idx:key_idx+1, ...] > 0.5)
                            print(f"L2 loss: {loss}, DICE loss: {dice_loss}")
                            loss = torch.add(loss, dice_loss.mean())
                        
                        # last_masks = estimate_masks_bin.squeeze(1) > 0.5
                        last_masks = estimate_masks_bin.squeeze(1)
                        del estimate_masks, estimate_masks_bin 

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                    train_loss += avg_loss.item() / gradient_accumulation_steps

                    
                    # Run backpropagation
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)

                    optimizer.step()                
                    optimizer.zero_grad()

                    # Learning rate scheduler step
                    lr_scheduler.step()
                    progress_bar.update(1)
                    global_step += 1

                
                # keep other embeddings unchanged:
                if word_to_learn is not None:
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    index_no_updates[min(added_token_ids) : max(added_token_ids) + 1] = False

                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                return latents, loss, loss_logger
            
            with accelerator.accumulate(unet):
                '''
                log 240814: added `accelerator.accumulate(text_encoder)` for learnable textual-emb setting.
                Not sure if this 100% fits original code - pending
                '''
                with accelerator.accumulate(text_encoder):
                    latents, loss, loss_logger = train_step(train_loss, loss_logger)     
            
            # Wandb logging
            # Check if the accelerator has performed an optimization step behind the scenes
            # if accelerator.sync_gradients:
            #     progress_bar.update(1)
            #     global_step +=  1
            #     accelerator.log({"train_loss":train_loss},step=global_step)

            if global_step % 100 == 0:
                if log_losses_in_csv:
                    loss_logger.to_csv(f'./{output_dir}/losses.csv', index=False)
                    
            # Save checkpoint
            if accelerator.is_main_process and (global_step % checkpointing_steps == 0):
                save_path = os.path.join(output_dir, f"checkpoints_{global_step}")
                os.makedirs(save_path, exist_ok=True)
                if ckpt_mode == 'split': 
                    unet_state_dict = unet.state_dict()
                    lora_state_dict = {}
                    i2v_state_dict = {}
                    ca_lora_state_dict = {}
                    for k in unet_state_dict.keys():
                        if "lora" in k and "attn2" not in k:
                            lora_state_dict[k] = unet_state_dict[k]
                        if "i2v" in k:
                            i2v_state_dict[k] = unet_state_dict[k]
                        if "attn2" in k and "lora" in k:
                            ca_lora_state_dict[k] = unet_state_dict[k]

                    torch.save(i2v_state_dict,save_path + "/animatediff-i2v-nv1-motionlora-i2v-weights.pt")
                    torch.save(lora_state_dict,save_path + "/animatediff-i2v-nv1-motionlora-lora-weights.pt")
                    torch.save(ca_lora_state_dict,save_path + "/animatediff-i2v-nv1-ca-lora-weights.pt")
                    print(f"{sum([len(i2v_state_dict), len(lora_state_dict), len(ca_lora_state_dict)])} parameters in total saved into 3 separate files.")
                elif ckpt_mode == 'adapter':
                    unet_state_dict = unet.state_dict()
                    counter = 0
                    adapter_state_dict = {}
                    for k in unet_state_dict.keys():
                        if any([name in k for name in trainable_modules]):
                            adapter_state_dict[k] = unet_state_dict[k]
                            counter += 1
                    torch.save(adapter_state_dict, save_path + "/adapter.pt") # save to {ckpt_iter}/adapter.pt
                    print(f"{len(adapter_state_dict)} parameters saved.")
                else:
                    accelerator.save_state(save_path)
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
                # saving new embeddings ckpt at each saving global step
                if word_to_learn is not None:
                    weight_name = f"learned_embeds-steps-{global_step}.{textemb_file_format}"
                    save_path = os.path.join(output_dir, weight_name)
                    save_progress(
                        text_encoder,
                        accelerator,
                        word_to_learn,
                        save_path,
                        safe_serialization = textemb_file_format=="safetensors",
                    )
                
            # Periodically validation
            if accelerator.is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.height
                width  = train_data.width

                prompts = validation_data.prompt_path
                video_length = validation_data.video_length
                use_mask_branch = unet_additional_kwargs.get("use_mask_branch", False)
                mask_branch_kwargs = unet_additional_kwargs.get("mask_branch_kwargs", dict())
                use_mask_guidance = mask_branch_kwargs.get("use_mask_guidance", False)
                blurring = mask_branch_kwargs.get("blurring", False)
                mask_guided_attention = mask_branch_kwargs.get("mask_guided_attention", "SA")
                for idx, prompt in enumerate(prompts):
                    if use_Mask_data:
                        image = Image.open(os.path.join(validation_data.image_path, prompt.replace(' ', '_') + '_mask.jpg')).convert("RGB")    
                    elif use_DAVIS_data:
                        image = Image.open(os.path.join(validation_data.image_path, prompt.replace(' ', '_') + '.jpg'))
                    else:
                        image = Image.open(os.path.join(validation_data.image_path, prompt.replace(' ', '_') + '.png'))
                    image = np.asarray(image)

                    # if use_mask_guidance:
                    #     masks = torch.stack([torch.tensor(np.array(Image.open(os.path.join(validation_data.mask_path, prompt.replace(" ", '_'), f"{i + 1}_mask.jpg")).resize((width // 8, height // 8)).convert("L")) > 127, dtype=torch.float16, device="cuda") for i in range(video_length)])
                    #     attention_mask = interframe_mask(masks, video_length, unet_additional_kwargs.get("sa_mode", "first"), device="cuda")

                    # Remove alpha channel if existent
                    if len(image.shape) == 3 and image.shape[2] == 4:
                        image = image[:, :, : 3]

                    if use_mask_branch:
                        image_m = Image.open(os.path.join(validation_data.mask_path, prompt.replace(" ", '_') + "_mask.jpg")).convert("RGB")
                        image_m = np.asarray(image_m)
                        if len(image_m.shape) == 3 and image_m.shape[2] == 4:
                            image_m = image_m[:, :, : 3]

                    # Restore RGB colors
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

                    if do_clip_image_emb:
                        frame = image_processor(images=image / 255, return_tensors="pt", do_rescale=False)["pixel_values"].to("cuda")
                        clip_image_embeds = image_encoder(frame).image_embeds
                    else:
                        clip_image_embeds = None

                    first_frame_latent = torch.Tensor(image.copy()).to(latents.device).type_as(latents).permute(2, 0, 1).repeat(1, 1, 1, 1)
                    first_frame_latent = first_frame_latent / 127.5 - 1.0
                    first_frame_latent = vae.encode(first_frame_latent).latent_dist.sample() * 0.18215
                    first_frame_latent = first_frame_latent.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
                    
                    if hasattr(validation_data, "first_frame_cond_type") and validation_data.first_frame_cond_type == "cinemo":
                        first_frame_latent = repeat(first_frame_latent, 'b c f h w -> b c (f r) h w', r=15).contiguous()

                    if use_mask_branch:
                        image_m = cv2.resize(image_m, (width, height))
                        first_mask_latent = torch.Tensor(image_m.copy()).to(latents.device).type_as(latents).permute(2, 0, 1).repeat(1, 1, 1, 1)
                        first_mask_latent = first_mask_latent / 127.5 - 1.0
                        first_mask_latent = vae.encode(first_mask_latent).latent_dist.sample() * 0.18215
                        first_mask_latent = first_mask_latent.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
                        
                        if hasattr(validation_data, "first_frame_cond_type") and validation_data.first_frame_cond_type == "cinemo":
                            first_mask_latent = repeat(first_mask_latent, 'b c f h w -> b c (f r) h w', r=15).contiguous()

                    if use_DDIMInv_data:
                        validation_data.first_frame_cond_type = "inv"
                
                    if not image_finetune:
                        if use_ip_adapter == "IP-Adapter": 
                            clip_image = ip_adapter.clip_image_processor(images = image,return_tensors = "pt").pixel_values
                            clip_image_embeds = ip_adapter.image_encoder(clip_image.to(torch.device("cuda"),dtype=torch.float16)).image_embeds
                            image_prompt_embeds = ip_adapter.image_proj_model(clip_image_embeds)   
                            uncond_image_prompt_embeds = ip_adapter.image_proj_model(torch.zeros_like(clip_image_embeds))
                            sample = validation_pipeline(
                                prompt= validation_data.prompts[idx],
                                image_prompt_embeds = image_prompt_embeds, 
                                uncond_image_prompt_embeds=uncond_image_prompt_embeds, 
                                generator    = generator,
                                latents = first_frame_latent,
                                repeat_token_mode = repeat_token_mode,
                                **validation_data,
                            ).videos
                            save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                            samples.append(sample)
                        else:
                            image_prompt_embeds = None
                            uncond_image_prompt_embeds = None

                            sample = validation_pipeline(
                                prompt= validation_data.prompts[idx], 
                                image_prompt_embeds = image_prompt_embeds, 
                                uncond_image_prompt_embeds=uncond_image_prompt_embeds, 
                                generator = generator,
                                latents = first_frame_latent,
                                repeat_token_mode = repeat_token_mode,
                                ddim_inv_latent = torch.load(os.path.join(validation_data.DDIMInv_latent_path, prompt.replace(' ', '_') + '.pt')) if use_DDIMInv_data else None,
                                mask_latents = first_mask_latent if use_mask_branch else None,
                                use_mask_guidance = use_mask_guidance,
                                blurring = blurring,
                                mask_inference_stride = 1,
                                mask_inference_cutoff = 50,
                                use_mask_branch = use_mask_branch,
                                mask_guided_attention = mask_guided_attention,
                                sa_mode = unet_additional_kwargs.get("sa_mode", "first"),
                                image_embeds = clip_image_embeds,
                                **validation_data,
                            ).videos

                            save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                            samples.append(sample)
                        
                    else:
                        sample = validation_pipeline(
                            validation_data.prompts[idx],
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                            guidance_scale      = validation_data.get("guidance_scale", 8.),
                            repeat_token_mode   = repeat_token_mode
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)
                
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
        # print(f"1: {global_step} 2: {max_train_steps}")
            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break
            
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if word_to_learn is not None:
            text_encoder = accelerator.unwrap_model(text_encoder)

        # Save the newly trained embeddings
        if word_to_learn is not None:
            weight_name = f"learned_embeds.{textemb_file_format}"
            save_path = os.path.join(output_dir, weight_name)
            save_progress(
                text_encoder,
                accelerator,
                word_to_learn,
                save_path,
                safe_serialization = textemb_file_format=="safetensors",
            )
    accelerator.end_training()


if __name__ == "__main__":
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(**config)
