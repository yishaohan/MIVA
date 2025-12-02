
import os
import cv2
import argparse
import glob
import re

from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Iterable, Tuple
from PIL import Image
from einops import repeat

import numpy as np
import torch
import torchvision
import time

from diffusers import AutoencoderKL, DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.models.ptp_utils import AttentionStore, show_cross_attention, display_image, show_temporal_CA, show_simple_cross_attention, show_temporal_attention

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument('--format', type=str, default='gif', help='output file format (gif/mp4/frames)')
parser.add_argument('--save_samples', action="store_true")
parser.set_defaults(save_samples=False)


def load_checkpoints(
    unet_checkpoint_path: str = "",
    ckpt_mode: str = "", 
    adapter_i2v_path: str = "",
    adapter_lora_path: str = "",
    ca_lora_path: str = "",
    unet_additional_kwargs: Dict = {},
    load_from_old_smplCA_ckpt = False,
):
    '''
    load_from_adapter_ckpt: new feature. See the same item in configs/inference/temp_multi_ca.yaml
    '''
    state_dict = {}
    if ckpt_mode == 'split':
        if isinstance(adapter_i2v_path, str):
            num_adapters = 1
        elif isinstance(adapter_i2v_path, Iterable) and len(adapter_i2v_path)==1:
            num_adapters = 1
            adapter_i2v_path = adapter_i2v_path[0]
            adapter_lora_path = adapter_lora_path[0]
            ca_lora_path = ca_lora_path[0]
        else:
            num_adapters = len(adapter_i2v_path)
    elif ckpt_mode == 'adapter':
        if isinstance(unet_checkpoint_path, str):
            num_adapters = 1
        elif isinstance(unet_checkpoint_path, Iterable) and len(unet_checkpoint_path)==1:
            num_adapters = 1
            unet_checkpoint_path = unet_checkpoint_path[0]
        else:
            num_adapters = len(unet_checkpoint_path)
    else: 
        if isinstance(unet_checkpoint_path,str) and unet_checkpoint_path!="":
            num_adapters = 1
        elif isinstance(unet_checkpoint_path, Iterable) and len(unet_checkpoint_path)==1:
            num_adapters = 1
            unet_checkpoint_path = unet_checkpoint_path[0]
        else:
            num_adapters = len(unet_checkpoint_path)    
    unet_additional_kwargs["num_adapters"] = num_adapters

    if "adapter_weights" not in unet_additional_kwargs.keys():
        unet_additional_kwargs["adapter_weights"] = [1] * num_adapters # log 240724: default weight set as 1.
    
    # Load pretrained unet weights
    if ckpt_mode ==  'split':
        print("loading from split checkpoint")
        if isinstance(adapter_i2v_path, str):
            old_adapter_state_dict = torch.load(adapter_i2v_path)
            old_adapter_state_dict = old_adapter_state_dict["state_dict"] if "state_dict" in old_adapter_state_dict else old_adapter_state_dict
            for k in old_adapter_state_dict.keys():
                state_dict[k] = old_adapter_state_dict[k]

            old_lora_state_dict = torch.load(adapter_lora_path)
            old_lora_state_dict = old_lora_state_dict["state_dict"] if "state_dict" in old_lora_state_dict else old_lora_state_dict
            for k in old_lora_state_dict.keys():
                state_dict[k] = old_lora_state_dict[k]

            old_ca_lora_state_dict = torch.load(ca_lora_path)
            old_ca_lora_state_dict = old_ca_lora_state_dict["state_dict"] if "state_dict" in old_ca_lora_state_dict else old_ca_lora_state_dict
            for k in old_ca_lora_state_dict.keys():
                state_dict[k] = old_ca_lora_state_dict[k]

        else:
            for i, i2v_path in enumerate(adapter_i2v_path):
                # load fine-tuned weights from 3 separate ckpt files
                old_adapter_state_dict = torch.load(i2v_path)
                old_adapter_state_dict = old_adapter_state_dict["state_dict"] if "state_dict" in old_adapter_state_dict else old_adapter_state_dict
                for key in old_adapter_state_dict.keys():
                    if "i2v_adapter" in key:
                        new_key = key.replace("i2v_adapter.", "i2v_adapters.{}.".format(i))
                    else: 
                        new_key = key          
                    state_dict[new_key] = old_adapter_state_dict[key]
                
                old_lora_state_dict = torch.load(adapter_lora_path[i])
                old_lora_state_dict = old_lora_state_dict["state_dict"] if "state_dict" in old_lora_state_dict else old_lora_state_dict
                for key in old_lora_state_dict.keys():
                    
                    if ("processor") in key and ("_ip" not in key) and ("attn2" not in key):
                        new_key = key.replace("processor.", "processor.processors.{}.".format(i))
                    else: 
                        new_key = key
                    state_dict[new_key] = old_lora_state_dict[key]

                old_ca_lora_state_dict = torch.load(ca_lora_path[i])
                old_ca_lora_state_dict = old_ca_lora_state_dict["state_dict"] if "state_dict" in old_ca_lora_state_dict else old_ca_lora_state_dict
                for key in old_ca_lora_state_dict.keys():
                    if "attn2" in key:
                        new_key = key.replace("to_k_lora.", "to_k_loras.{}.".format(i))
                        new_key = new_key.replace("to_v_lora.", "to_v_loras.{}.".format(i))
                        new_key = new_key.replace("to_out_lora.", "to_out_loras.{}.".format(i))
                    else: 
                        new_key = key              
                    state_dict[new_key] = old_ca_lora_state_dict[key]
    elif ckpt_mode == "adapter": 
        # log 250224: build state_dict from adapter PT.
        print("loading from adapter checkpoint")
        if isinstance(unet_checkpoint_path, str):
            old_adapter_state_dict = torch.load(unet_checkpoint_path)
            old_adapter_state_dict = old_adapter_state_dict["state_dict"] if "state_dict" in old_adapter_state_dict else old_adapter_state_dict
            for k in old_adapter_state_dict.keys():
                new_k = k
                # with open("./temp/param_name.txt", "a") as file:
                #     file.write(f"{k}\n")
                if load_from_old_smplCA_ckpt:
                    keywords = ["CA_norm", "CA2", "CA3"]
                    for word in keywords:
                        new_k = new_k.replace(f"{word}", f"sCA.{word}")
                state_dict[new_k] = old_adapter_state_dict[k]
        else:
            for i, adapter_path in enumerate(unet_checkpoint_path):
                # load fine-tuned weights from 3 separate ckpt files
                old_adapter_state_dict = torch.load(adapter_path)
                old_adapter_state_dict = old_adapter_state_dict["state_dict"] if "state_dict" in old_adapter_state_dict else old_adapter_state_dict
                for key in old_adapter_state_dict.keys():
                    if ("processor") in key and ("_ip" not in key) and ("attn2" not in key):
                        new_key = key.replace("processor.", "processor.processors.{}.".format(i))
                    else: 
                        new_key = key

                    if "i2v_adapter" in new_key:
                        new_key = key.replace("i2v_adapter.", "i2v_adapters.{}.".format(i))
                    else: 
                        new_key = new_key

                    if "attn2" in key: # CA-LoRA
                        new_key = new_key.replace("to_k_lora.", "to_k_loras.{}.".format(i))
                        new_key = new_key.replace("to_v_lora.", "to_v_loras.{}.".format(i))
                        new_key = new_key.replace("to_out_lora.", "to_out_loras.{}.".format(i))
                    elif "CA2" in key or "CA3" in key or "CA_norm" in key: # CA-LoRA
                        # print("reach")
                        new_key = new_key.replace("CA2.", "CA2.{}.".format(i))
                        new_key = new_key.replace("CA3.", "CA3.{}.".format(i))
                        new_key = new_key.replace("CA_norm.", "CA_norm.{}.".format(i))
                    else: 
                        new_key = new_key

                    state_dict[new_key] = old_adapter_state_dict[key]
    else: 
        print("loading from full checkpoint")
        if not isinstance(unet_checkpoint_path, str) or unet_checkpoint_path != "":
            if isinstance(unet_checkpoint_path, str):
                with safe_open(unet_checkpoint_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            else:
                for i, unet_path in enumerate(unet_checkpoint_path):
                    with safe_open(unet_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if ("processor") in key and ("_ip" not in key) and ("attn2" not in key):
                                new_key = key.replace("processor.", "processor.processors.{}.".format(i))
                            else: 
                                new_key = key
                            if "i2v_adapter" in new_key:
                                new_key = new_key.replace("i2v_adapter.", "i2v_adapters.{}.".format(i))
                            else: 
                                new_key = new_key
                            if "attn2" in key: # CA-LoRA
                                new_key = new_key.replace("to_k_lora.", "to_k_loras.{}.".format(i))
                                new_key = new_key.replace("to_v_lora.", "to_v_loras.{}.".format(i))
                                new_key = new_key.replace("to_out_lora.", "to_out_loras.{}.".format(i))
                            else: 
                                new_key = new_key
                            state_dict[new_key] = f.get_tensor(key)

    print(f"[load_ckpt]Loaded {len(state_dict)} parameters.")
    return state_dict, unet_additional_kwargs


def main(
    output_dir: str,
    pretrained_model_path: str,
    motion_module_path: str,
    use_ip_adapter: str,
    validation_data: Dict,
    ip_ckpt: str = None,
    image_encoder_path: str = None,
    repeat_token_mode: str = None,
    unet_checkpoint_path: str = "",
    textemb_checkpoint_path: str = "",
    adapter_i2v_path: str = "",
    adapter_lora_path: str = "",
    ca_lora_path: str = "", 
    unet_additional_kwargs: Dict = {},
    noise_scheduler_kwargs = None,
    save_samples: bool = False,
    format: str = "gif",
    global_seed: int = 42,
    ckpt_mode: str = "",
    use_DDIMInv_data = False,
    use_Mask_data = False,
    load_from_old_smplCA_ckpt = False,
    pretrained_sa_adaptive_weight_module = None,
):
    print("global_seed:", global_seed)
    torch.manual_seed(global_seed)
    weight_dtype=torch.float16
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(dtype=weight_dtype)
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    do_clip_image_emb = unet_additional_kwargs.get("motion_module_kwargs", {}).get("simple_CA_kwargs", {}).get("use_clip_image_emb", False)
    if do_clip_image_emb:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
            "cuda", dtype=torch.float32
        )
        image_processor = CLIPImageProcessor()

    state_dict, unet_additional_kwargs = load_checkpoints(
        unet_checkpoint_path, ckpt_mode, adapter_i2v_path, adapter_lora_path, ca_lora_path, unet_additional_kwargs, load_from_old_smplCA_ckpt)

    unet_additional_kwargs["use_ip_adapter"] = use_ip_adapter
    width = validation_data["width"]
    height = validation_data["height"]
    unet_additional_kwargs["video_resolution"] = (width, height)
    
    
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path,
        motion_module_path,
            subfolder="unet", 
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
    ).to(dtype=weight_dtype)

    controller = None
    if validation_data.get("CA_Map_visualization_resolution", None) or validation_data.get("TA_Map_visualization_resolution", None):
        controller = AttentionStore()
        if validation_data.get("CA_Map_visualization_resolution", None):
            print("[Inference]CA Map visualization: AttentionStore created.")
        else:
            print("[Inference]TA Map visualization: AttentionStore created.")

    tCA_controller = None
    if validation_data.get("temporal_CA_Map_visualization_resolution", None):
        tCA_controller = AttentionStore()
        print("[Inference]temporal CA Map Visualization: AttentionStore created.")

    ## copy SA QKV
    # For I2V-Adapter, W_Q/K/V are copied from the base UNet.
    if ckpt_mode == 'adapter':
        with torch.no_grad():
            for name, param in unet.named_parameters():
                if 'i2v_adapter' in name and 'lora' not in name and 'to_out' not in name and "weight_module" not in name:
                    source_w = name.replace("to_q_mask", "to_q")
                    source_w = source_w.replace("to_out_mask", "to_out")
                    source_w = source_w.replace("processor.", "")
                    if unet_additional_kwargs["num_adapters"] > 1:
                        source_w = re.sub(r"i2v_adapters\.\d+\.", "attn1.", name)
                    else:
                        source_w = name.replace('i2v_adapter', 'attn1')
                    param.copy_(unet.state_dict()[source_w])
                if 'CA1' in name:
                    source_w = name.replace("sCA.CA1", "attn2.to_q")
                    source_w = source_w.replace("sCA_b.CA1", "attn2.to_q")
                    param.copy_(unet.state_dict()[source_w])
                if 'CA4' in name:
                    source_w = name.replace("sCA.CA4", "attn2.to_out.0")
                    source_w = source_w.replace("sCA_b.CA4", "attn2.to_out.0")
                    param.copy_(unet.state_dict()[source_w]) 
            
    m, u = unet.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    for k in m:
        with open("./temp/missing_param_name.txt", "a") as file:
                    file.write(f"{k}\n")

    if use_ip_adapter == "IP-Adapter" or use_ip_adapter == "ID-Animator":
        from ip_adapter import IPAdapter 
        ip_adapter = IPAdapter(unet, image_encoder_path, ip_ckpt, torch.device("cuda"), use_id_animator = use_ip_adapter == "ID-Animator")
        image_proj_model = ip_adapter.image_proj_model

        # Load the IP-Adapter checkpoint
        ip_state_dict = torch.load(ip_ckpt, map_location="cpu")
        image_proj_model.load_state_dict(ip_state_dict["image_proj"])

        # Get IP layers from unet
        ip_layers = {**unet.find_layers(keyword="to_k_ip"), **unet.find_layers(keyword="to_v_ip")}

        # Sort layers
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

        sorted_ip_layers = sorted(ip_layers.items(), key=lambda item: layer_sort_key(item[0]))
        sorted_weight_keys = sorted(ip_state_dict['ip_adapter'].keys(), key=lambda key: int(key.split('.')[0]))

        # Load weights into unet's IP layers
        for (layer_name, layer_module), weight_key in zip(sorted_ip_layers, sorted_weight_keys):
            weight_tensor = ip_state_dict['ip_adapter'][weight_key]
            if layer_module.weight.shape == weight_tensor.shape:
                with torch.no_grad():
                    layer_module.weight.copy_(weight_tensor)

        print("IP adapter initialized")

    # ablation study - load frozen SA adaptive weight module from an existing ckpt
    if pretrained_sa_adaptive_weight_module is not None:
        from scripts.load_sa_adaptive_weight_module import load_sa_adaptive_weight_module
        load_sa_adaptive_weight_module(unet, pretrained_sa_adaptive_weight_module)
    
    validation_pipeline = AnimationPipeline(
        unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, attention_store = controller, tCA_attention_store = tCA_controller
    ).to("cuda")

    if validation_data.get("word_to_replace", None):
        '''
        log 240826:
        UNet ckpt at <unet_path>/checkpoints/model.safetensors
        Text-emb at <unet_path>/learned_embeds.safetensors
        '''
        if isinstance(validation_data.word_to_replace, str):
            ti_ckpt_path = os.path.dirname(os.path.dirname(unet_checkpoint_path))
            print(f"TI Checkpoint path: {ti_ckpt_path}")
            validation_pipeline.load_textual_inversion(ti_ckpt_path)
            print("TI embeddings loaded!")
        elif isinstance(validation_data.word_to_replace, Iterable) and isinstance(unet_checkpoint_path, Iterable):
            for i in range(len(unet_checkpoint_path)):
                ti_ckpt_path = os.path.dirname(os.path.dirname(unet_checkpoint_path[i]))
                validation_pipeline.load_textual_inversion(ti_ckpt_path)
                print("{} TI embeddings loaded!".format((i+1)))
    
    samples = []
    generator = torch.Generator(device=unet.device)
    generator.manual_seed(global_seed)
    prompts = validation_data.prompt_path # the input images

    # log 240807: some special settings for prompt_path
    # for "all" / "newlamp", users must use "batch_prompt" to assign the prompt.
    if prompts == "all":
        if use_Mask_data:
            image_paths = glob.glob(f"{validation_data.image_path}/*_mask.jpg")
        else:
            image_paths = glob.glob(f"{validation_data.image_path}/*.png")
    elif prompts == "newlamp":
        if use_Mask_data:
            image_paths = [f"{validation_data.image_path}/{i}_mask.jpg" for i in range(1,21)]
        else:
            image_paths = [f"{validation_data.image_path}/{i}.png" for i in range(1,21)]
    elif isinstance(prompts, Iterable):
        image_paths = prompts.copy()
        for i, prompt in enumerate(prompts):
            img_name = prompt.replace(' ', '_')
            if use_Mask_data:
                image_paths[i] = os.path.join(validation_data.image_path, img_name + '_mask.jpg')
            else:
                image_paths[i] = os.path.join(validation_data.image_path, img_name + '.png')
    else:
        raise NotImplementedError
    
    if use_DDIMInv_data:
        # assume we have prepared the inversion result "n.pt" under <validation_data.DDIMInv_latent_path> for the input image "n.png"
        if prompts == "add":
            latent_paths = glob.glob(f"{validation_data.DDIMInv_latent_path}/*.pt")
        elif prompts == "newlamp":
             latent_paths = [f"{validation_data.DDIMInv_latent_path}/{i}.pt" for i in range(1,21)]
        elif isinstance(prompts, Iterable):
            latent_paths = prompts.copy()
            for i, prompt in enumerate(prompts):
                latent_name = prompt.replace(' ', '_')
                latent_paths[i] = os.path.join(validation_data.DDIMInv_latent_path, latent_name + '.pt')
        else:
            raise NotImplementedError

    width = validation_data.get("width", 512)
    height = validation_data.get("height", 320)

    mask_branch_kwargs = unet_additional_kwargs.get("mask_branch_kwargs", dict())
    use_mask_branch = unet_additional_kwargs.get("use_mask_branch", False)
    use_mask_guidance = mask_branch_kwargs.get("use_mask_guidance", False)
    mask_inference_stride = unet_additional_kwargs.get("mask_inference_stride", 1)
    mask_inference_cutoff = unet_additional_kwargs.get("mask_inference_cutoff", 50)
    blurring = mask_branch_kwargs.get("blurring", False)
    for idx, path in enumerate(image_paths):
        start = time.perf_counter()
        if use_Mask_data:
            image = Image.open(path).convert("RGB")
        else:
            image = Image.open(path)
        image = np.asarray(image)
        img_name = os.path.basename(path)[:-4]

        video_length = validation_data.video_length
        
        # # Remove alpha channel if existent
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, : 3]

        if use_mask_branch:
            if isinstance(validation_data.mask_path, str):
                image_m = Image.open(os.path.join(validation_data.mask_path, img_name + "_mask.jpg")).convert("RGB")
                image_m = np.asarray(image_m)
                if len(image_m.shape) == 3 and image_m.shape[2] == 4:
                    image_m = image_m[:, :, : 3]
            else:
                image_m = []
                for i, path in enumerate(validation_data.mask_path):
                    image_m.append(Image.open(os.path.join(path, img_name + "_mask.png")).convert("RGB"))
                    image_m[i] = np.asarray(image_m[i])
                    if len(image_m[i].shape) == 3 and image_m[i].shape[2] == 4:
                        image_m[i] = image_m[i][:, :, : 3]

        # # Restore RGB colors
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        if do_clip_image_emb:
            frame = image_processor(images=image / 255, return_tensors="pt", do_rescale=False)["pixel_values"].to("cuda")
            clip_image_embeds = image_encoder(frame).image_embeds

        first_frame_latent = torch.Tensor(image.copy()).to(unet.device).to(weight_dtype).permute(2, 0, 1).repeat(1, 1, 1, 1)
        first_frame_latent = first_frame_latent / 127.5 - 1.0
        first_frame_latent = vae.encode(first_frame_latent).latent_dist.sample() * 0.18215
        first_frame_latent = first_frame_latent.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)

        if hasattr(validation_data, "first_frame_cond_type") and validation_data.first_frame_cond_type == "cinemo":
            first_frame_latent = repeat(first_frame_latent, 'b c f h w -> b c (f r) h w', r=15).contiguous()

        if use_mask_branch:
            if isinstance(image_m, list):
                first_mask_latent = []
                for i, img in enumerate(image_m):
                    img = cv2.resize(img, (width, height))
                    first_mask_latent.append(torch.Tensor(img.copy()).to(unet.device).to(weight_dtype).permute(2, 0, 1).repeat(1, 1, 1, 1))
                    first_mask_latent[i] = first_mask_latent[i] / 127.5 - 1.0
                    first_mask_latent[i] = vae.encode(first_mask_latent[i]).latent_dist.sample() * 0.18215
                    first_mask_latent[i] = first_mask_latent[i].repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
                    
                    if hasattr(validation_data, "first_frame_cond_type") and validation_data.first_frame_cond_type == "cinemo":
                        first_mask_latent[i] = repeat(first_mask_latent[i], 'b c f h w -> b c (f r) h w', r=15).contiguous()    
            else:
                image_m = cv2.resize(image_m, (width, height))
                first_mask_latent = torch.Tensor(image_m.copy()).to(unet.device).to(weight_dtype).permute(2, 0, 1).repeat(1, 1, 1, 1)
                first_mask_latent = first_mask_latent / 127.5 - 1.0
                first_mask_latent = vae.encode(first_mask_latent).latent_dist.sample() * 0.18215
                first_mask_latent = first_mask_latent.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
                
                if hasattr(validation_data, "first_frame_cond_type") and validation_data.first_frame_cond_type == "cinemo":
                    first_mask_latent = repeat(first_mask_latent, 'b c f h w -> b c (f r) h w', r=15).contiguous()

        if hasattr(validation_data, "batch_prompt"):
            prompt = validation_data.batch_prompt
            if isinstance(prompt, Iterable):
                prompt = list(prompt)
                if len(prompt)==1:
                    prompt = prompt[0]
        elif isinstance(prompts, Iterable):
            prompt = validation_data.prompts[idx]
        else:
            raise NotImplementedError
    
        if use_ip_adapter == "ID-Animator": 
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            from insightface.utils import face_align
            img = cv2.imread(path)
            faces = app.get(img)
            face_roi = face_align.norm_crop(img,faces[0]['kps'],112)
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_image = [Image.fromarray(face_roi).resize((224, 224))]

            clip_image = ip_adapter.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(torch.device("cuda"), dtype=torch.float16)
            clip_image_embeds = ip_adapter.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = ip_adapter.image_proj_model(clip_image_embeds)
            uncond_clip_image_embeds = ip_adapter.image_encoder(
                torch.zeros_like(clip_image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds = ip_adapter.image_proj_model(uncond_clip_image_embeds)
        elif use_ip_adapter == "IP-Adapter":
            clip_image = ip_adapter.clip_image_processor(images = image, return_tensors = "pt").pixel_values
            clip_image_embeds = ip_adapter.image_encoder(clip_image.to(torch.device("cuda"), dtype=torch.float16)).image_embeds
            image_prompt_embeds = ip_adapter.image_proj_model(clip_image_embeds)
            uncond_image_prompt_embeds = ip_adapter.image_proj_model(torch.zeros_like(clip_image_embeds))
        else:
            image_prompt_embeds = None
            uncond_image_prompt_embeds = None

        if use_DDIMInv_data:
            validation_data.first_frame_cond_type = "inv"

        video = validation_pipeline(
            prompt = prompt,
            generator = generator,
            latents = first_frame_latent,
            image_prompt_embeds = image_prompt_embeds,
            uncond_image_prompt_embeds = uncond_image_prompt_embeds,
            is_inference = True,
            repeat_token_mode = repeat_token_mode,
            ddim_inv_latent = torch.load(latent_paths[idx]) if use_DDIMInv_data else None,
            mask_latents = first_mask_latent if use_mask_branch else None,
            use_mask_branch = use_mask_branch,
            use_mask_guidance = use_mask_guidance,
            mask_inference_stride = mask_inference_stride,
            mask_inference_cutoff = mask_inference_cutoff,
            sa_mode = unet_additional_kwargs.get("sa_mode", "first"),
            blurring = blurring,
            image_embeds = clip_image_embeds,
            **validation_data,
        ).videos

        end = time.perf_counter()
        print(f"time of execution: {end - start:.4f} seconds")

        if validation_data.get("CA_Map_visualization_resolution", None):
            resolution = validation_data.get("CA_Map_visualization_resolution", None)
            if unet_additional_kwargs.get("q_downsample", False):
                q_downsample_ratio = unet_additional_kwargs["q_downsample_ratio"]
                print(f"STQ case, downsample by {q_downsample_ratio}")
            else:
                q_downsample_ratio = -1
                print("No STQ detected")
            
            if resolution == "all":
                print("Getting all resolutions")
                resolution = ["low", "middle", "high"]
            
            ca_map_path = os.path.join(output_dir, f"{img_name}_CAMap")
            os.makedirs(ca_map_path, exist_ok=True)
            for res in resolution:
                if res == "low":
                    width = 512 // (8 * 2 * 2)
                    height = 320 // (8 * 2 * 2)
                if res == "middle":
                    width = 512 // (8 * 2)
                    height = 320 // (8 * 2)
                if res == "high":
                    width = 512 // (8)
                    height = 320 // (8)
                if q_downsample_ratio != -1:
                    width = width // q_downsample_ratio
                    height = height // q_downsample_ratio
                output_path = os.path.join(ca_map_path, f"{'STQ_' if q_downsample_ratio!=-1 else ''}{res}")
                if unet_additional_kwargs.get("use_simple_CA", False):
                    show_simple_cross_attention(controller, width = width, height = height, output_path = output_path)
                else:
                    show_cross_attention(controller, width = width, height = height, q_downsample = q_downsample_ratio, output_path = output_path)

        if validation_data.get("TA_Map_visualization_resolution", None):
            resolution = validation_data.get("TA_Map_visualization_resolution", None)
            
            if resolution == "all":
                print("Getting all resolutions")
                resolution = ["low", "middle", "high"]
            
            ta_map_path = os.path.join(output_dir, f"{img_name}_TAMap")
            os.makedirs(ta_map_path, exist_ok=True)
            for res in resolution:
                if res == "low":
                    width = 512 // (8 * 2 * 2)
                    height = 320 // (8 * 2 * 2)
                if res == "middle":
                    width = 512 // (8 * 2)
                    height = 320 // (8 * 2)
                if res == "high":
                    width = 512 // (8)
                    height = 320 // (8)
                output_path = os.path.join(ta_map_path, f"{res}")
                
                show_temporal_attention(controller, width = width, height = height, output_path = output_path)
        
        if validation_data.get("temporal_CA_Map_visualization_resolution", None):
            resolution = validation_data.get("temporal_CA_Map_visualization_resolution", None)
            
            if resolution == "all":
                print("Getting all resolutions")
                resolution = ["low", "middle", "high"]
            
            ca_map_path = os.path.join(output_dir, f"{img_name}_temporal_CAMap")
            os.makedirs(ca_map_path, exist_ok=True)
            for res in resolution:
                if res == "low":
                    width = 512 // (8 * 2 * 2)
                    height = 320 // (8 * 2 * 2)
                if res == "middle":
                    width = 512 // (8 * 2)
                    height = 320 // (8 * 2)
                if res == "high":
                    width = 512 // (8)
                    height = 320 // (8)
                output_path = os.path.join(ca_map_path, f"{res}")
                show_temporal_CA(tCA_controller, width = width, height = height, output_path = output_path)

        if validation_data.get("hist_match", False):
            # apply histogram matching to output video
            from scripts.dip import hist_match
            print("Run Hist Match on frames.")
            for f in range(1, video.shape[2]):
                former_frame = video[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
                frame = video[0, :, f, :, :].permute(1, 2, 0).cpu().numpy()
                result = hist_match(former_frame, frame)
                result = torch.Tensor(result).type_as(video).to(video.device)
                video[0, :, f, :, :] = result.permute(2, 0, 1)

        if format=='images':
            video = (video.clamp(0, 1).squeeze().permute(1,2,3,0) * 255).to(torch.uint8) # ([16, 320, 512, 3])
            image_path = os.path.join(output_dir, img_name) # log 240822: for batch generation
            os.makedirs(image_path, exist_ok=True)
            index = 0
            for frame in video:
                frame = frame.cpu().numpy()
                frame = Image.fromarray(frame.astype(np.uint8))
                frame.save(os.path.join(image_path, f"{index}.png"))
                index += 1
            print("Saved images at:", image_path)
        elif format=='mp4':
            video = (video.clamp(0, 1).squeeze().permute(1,2,3,0) * 255).to(torch.uint8) # ([16, 320, 512, 3])
            save_path = os.path.join(output_dir, f"{img_name}.mp4")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.io.write_video(save_path, video, fps=8)
            print("Saved video:", save_path)
        elif format=='gif':
            save_path = os.path.join(output_dir, f"{img_name}.gif")
            save_videos_grid(video, save_path)
            print("Saved GIF:", save_path)
        elif format=='eval': # both MP4 & frames
            video = (video.clamp(0, 1).squeeze().permute(1,2,3,0) * 255).to(torch.uint8) # ([16, 320, 512, 3])
            save_path = os.path.join(output_dir, f"{img_name}.mp4")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.io.write_video(save_path, video, fps=8)
            print("Saved video:", save_path)

            image_path = os.path.join(output_dir, img_name)
            os.makedirs(image_path, exist_ok=True)
            index = 0
            for frame in video:
                frame = frame.cpu().numpy()
                frame = Image.fromarray(frame.astype(np.uint8))
                frame.save(os.path.join(image_path, f"{index}.png"))
                index += 1
            print("Saved images at:", image_path)
        else:
            raise ValueError("Invalid arg: format.")

    if save_samples:
        samples = torch.concat(samples)
        save_path = os.path.join(output_dir, "samples/sample-grid.gif")
        save_videos_grid(samples, save_path)

    print("Voila.")


if __name__ == "__main__":
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(format=args.format, save_samples=args.save_samples, **config)
