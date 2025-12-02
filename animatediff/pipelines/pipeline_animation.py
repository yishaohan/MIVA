# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union, Iterable
from dataclasses import dataclass
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import os
import torch
from tqdm import tqdm
from copy import deepcopy
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.loaders import TextualInversionLoaderMixin
try:
    from diffusers.pipeline_utils import DiffusionPipeline
except:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.models.seg_utils import gaussian_blur
from scripts.repeat_token import repeat_token
import torch.nn.functional as F
from .init_utils import *
from animatediff.utils.loss import estimate_x0
from torchvision.transforms.functional import rgb_to_grayscale
import torchvision

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[SparseControlNetModel, None] = None,
        attention_store = None,
        tCA_attention_store = None,
    ):
        super().__init__()
        self.attention_store = attention_store
        self.tCA_attention_store = tCA_attention_store
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    
    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt, image_prompt_embeds, uncond_image_prompt_embeds, word_to_replace=None, repeat_token_mode=None):
        '''
        log (240814): introduce new arg `word_to_replace` for learnable textual-emb setting.
        * if None, same as before;
        * if a str being ONE WORD, the new learned text Emb will replace the Emb of `word_to_replace`.
        '''

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if word_to_replace is not None and isinstance(word_to_replace, str):
            # find the token ID of `word_to_replace` in the vocabulary
            replace_id = self.tokenizer.encode(word_to_replace, add_special_tokens=False)[0]

            for i, id_input in enumerate(text_input_ids[0]):
                if id_input == replace_id:
                    text_input_ids[0][i] = len(self.tokenizer) - 1 # replace the ID; the learned textual-emb = last entry of the updated vocabulary
                    print("word(id) is replaced!")
                    break

        if repeat_token_mode is not None:
            text_embeddings = repeat_token(repeat_token_mode, text_input_ids, self.tokenizer, self.text_encoder)
        else:
            text_embeddings = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([text_embeddings,image_prompt_embeds],dim=1) if image_prompt_embeds is not None else text_embeddings
            uncond_embeddings = torch.cat([uncond_embeddings,uncond_image_prompt_embeds],dim=1) if uncond_image_prompt_embeds is not None else uncond_embeddings
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        else:
            text_embeddings = torch.cat([text_embeddings,image_prompt_embeds],dim=1) if image_prompt_embeds is not None else text_embeddings
        return text_embeddings
    
    def _encode_prompt_multi(self, prompt_list, device, num_videos_per_prompt, do_classifier_free_guidance, text_emb_dir=None, word_to_replace=None):
        '''
        Prepare text Embs under multi-adapter setting. Each adapter corresponds to fixed prompt (e.g. "waterfall").
        * text_emb_dir: if given, use presaved text Embs; otherwise, compute them online using CLIP text encoder.
            * for adapter A, we load "A.npy" under the folder
            * for unconditional Emb, we load "void.npy"

        Assumptions:
            * do_classifier_free_guidance = True
            * negative_prompt = ""
            * image_prompt_embeds = uncond_image_prompt_embeds = None
        Returns a text Emb Tensor, batch size = (len(word_list) + 1) * num_videos
                                                                  |- the unconditional word ""
            * Originally, it is 2 * num_videos_per_prompt, where former half = ""*num, latter half = prompt*num
                * use chunk(2) to separate them in CFG.
                * (240920) mind the signal order! In "multi" case it's in reversed order
        '''
        text_embeddings = []
        if text_emb_dir is None:
            for prompt in prompt_list:
                text_inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length)
                text_input_ids = text_inputs.input_ids

                if word_to_replace is not None and isinstance(word_to_replace, Iterable):
                    # find the token ID of `word_to_replace` in the vocabulary
                    # log(240829): for multiple adapters, 'word_to_replace' should be a list of str
                    replace_ids = {}
                    n_words_to_replace = len(word_to_replace)
                    for i in range(len(word_to_replace)):
                        replace_id = self.tokenizer.encode(word_to_replace[n_words_to_replace - 1 - i], add_special_tokens=False)[0]
                        # replace_ids.append(replace_id)
                        replace_ids[replace_id] = len(self.tokenizer) - i - 1 # loop from the end in reversed order
                        print(f"[TI] vocabulary link: \"{word_to_replace[n_words_to_replace - 1 - i]}\" {replace_id} => {replace_ids[replace_id]}")

                    for i, id_input in enumerate(text_input_ids[0]):
                        id_input = int(id_input)
                        if id_input in replace_ids:
                            # text_input_ids[0][i] = len(self.tokenizer) - (replace_ids.index(id_input) + 1) # replace the ID; the learned textual-emb = last entry of the updated vocabulary
                            text_input_ids[0][i] = replace_ids[id_input]
                            print(f"[TI] One of the words(ids) is replaced: {id_input} => {replace_ids[id_input]}")
                            break

                with torch.no_grad():
                    text_emb = self.text_encoder(text_input_ids.to(device))[0]
                    text_embeddings.append(text_emb)

            text_inputs = self.tokenizer("", return_tensors="pt", padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length)
            text_input_ids = text_inputs.input_ids
            with torch.no_grad():
                uncond_emb = self.text_encoder(text_input_ids.to(device))[0]
        else:
            for prompt in prompt_list:
                text_emb = torch.load(os.path.join(text_emb_dir, f"{prompt}.pt")).to(device)
                text_embeddings.append(text_emb)
            uncond_emb = torch.load(os.path.join(text_emb_dir, "void.pt")).to(device)

        text_embeddings = torch.cat(text_embeddings) # (prompt#, 77, 768)
        text_embeddings = text_embeddings.repeat(num_videos_per_prompt, 1, 1) # (prompt# * num, 77, 768)

        if do_classifier_free_guidance:
            uncond_embeddings = uncond_emb.repeat(num_videos_per_prompt, 1, 1) # (num, 77, 768)
            text_embeddings = torch.cat([text_embeddings, uncond_embeddings])

        return text_embeddings

    def decode_latents(self, latents, refl=False, normalize=False):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if normalize:
            video = (video - video.amin(dim=(1, 2, 3, 4), keepdim=True)) / \
            (video.amax(dim=(1, 2, 3, 4), keepdim=True) - video.amin(dim=(1, 2, 3, 4), keepdim=True) + 1e-8)
        if not refl:
            video = video.cpu().float().numpy()
        else:
            video = video.float()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, Iterable):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, analytic_init_timestep:int=None):
        # log (240904): analytic_init: if given (int), then initial noise is drawn from p_M() [CIL Eq.5]
        
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                if analytic_init_timestep is not None:
                    print("[Pipeline] CIL AnalyticInit triggered. Init timestep =", analytic_init_timestep)
                    from .init_utils import analytic_init
                    latents = analytic_init(self.scheduler, analytic_init_timestep, shape, generator, rand_device, dtype).to(device)
                else:
                    latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(dtype).to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image_prompt_embeds: torch.Tensor,
        uncond_image_prompt_embeds: torch.Tensor,
        video_length: Optional[int],
        repeat_token_mode: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        mask_latents = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        # support controlnet
        controlnet_images: torch.FloatTensor = None,
        controlnet_image_index: list = [0],
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        # inference tricks
        shared_noise_ratio = 0,
        first_frame_cond_type = "lamp",
        use_adain_norm: str = "off",
        ddim_inv_latent = None,
        use_dct_init:bool = False,
        dct_cutoff_ratio = 0.23,
        dct_cutoff_shape = 'rect',
        # multi_ca
        multi_ca: bool = False, # for multi-adapter setting only
        text_emb_dir = None,
        word_to_replace: Union[str, List[str]] = None,
        is_inference: bool = False, # a flag passed by inference.py. Only True when called by inference.py
        # CIL
        cil_ratio: float = 1,
        cil_analytic: bool = False,
        # demo 1-step x0 estimation (e.g., for ReFL)
        demo_x0_estimation_path: str = None, # if given, do x0 estimation and save the results here
        use_mask_branch = False,
        use_mask_guidance = False,
        mask_inference_stride = 1,
        mask_inference_cutoff = 50,
        blurring = False,
        mask_guided_attention = "SA",
        sa_mode = "first",
        **kwargs,
    ):
        '''
        use_adain_norm: "off" / "last" / "on"
            * on: apply AdaIN at every time step (setting of [LAMP])
            * last: apply AdaIN only at last time step (ours)
            * off(default): no AdaIN.

        multi_ca: if True, apply multiple K-Vs (per adapter);
        '''

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        mask_scheduler = deepcopy(self.scheduler)

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 

        if multi_ca:
            num_adapters = self.unet.num_adapters
            assert num_adapters == len(prompt), f"[Pipeline_Multi-CA mode]# of prompts should be # of adapters, prompt={prompt}."
            text_embeddings = self._encode_prompt_multi(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance,
                text_emb_dir=text_emb_dir, word_to_replace=word_to_replace, repeat_token_mode=repeat_token_mode
            )
        else: # original, outputs (2, 77, 768) for CFG. Further expanded to (2F, 77, 768) by Transformer3DModel.forward().
            text_embeddings = self._encode_prompt(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt,
                image_prompt_embeds, uncond_image_prompt_embeds, word_to_replace, repeat_token_mode=repeat_token_mode
            )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        mask_scheduler.set_timesteps(num_inference_steps // mask_inference_stride, device=device)
        timesteps = self.scheduler.timesteps
        # log (240902): Use smaller initial timestep [Conditional image leakage]
        if cil_ratio!=1:
            print("[Pipeline] CIL_ratio:", cil_ratio)
            timesteps = torch.round(timesteps * cil_ratio).int()

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        analytic_init_timestep = timesteps[0] if cil_ratio!=1 and cil_analytic else None
        if ddim_inv_latent is not None:
            noise_latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length * (1 if mask_latents is None else np.sum(use_mask_branch) + 1),
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                ddim_inv_latent,
                analytic_init_timestep,
            )
        else:
            noise_latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length * (1 if mask_latents is None else np.sum(use_mask_branch) + 1),
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                None,
                analytic_init_timestep,
            )
        if first_frame_cond_type == "lamp":
            print("[Pipeline] Noise Init mode:", first_frame_cond_type)
            first_frame_latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                1,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents[:, :, -1:, :, :],
            )
            if mask_latents is not None:
                if isinstance(mask_latents, list):
                    first_mask_latents = []
                    for i, mask_latent in enumerate(mask_latents):
                        first_mask_latent = self.prepare_latents(
                            batch_size * num_videos_per_prompt,
                            num_channels_latents,
                            1,
                            height,
                            width,
                            text_embeddings.dtype,
                            device,
                            generator,
                            mask_latent[:, :, -1:, :, :],
                        )
                        first_mask_latents.append(first_mask_latent)
                else:
                    first_mask_latents = self.prepare_latents(
                        batch_size * num_videos_per_prompt,
                        num_channels_latents,
                        1,
                        height,
                        width,
                        text_embeddings.dtype,
                        device,
                        generator,
                        mask_latents[:, :, -1:, :, :],
                    )

            if shared_noise_ratio > 0:
                print("[Pipeline]Shared noise ratio =", shared_noise_ratio)
                for f in range(1, video_length):
                    noise_latents[:, :, f:f+1, :, :] = shared_noise_ratio * noise_latents[:, :, 0:1, :, :] +\
                        (1-shared_noise_ratio) * noise_latents[:, :, f:f+1, :, :]
                if mask_latents is not None:
                    for s in range(1, np.sum(use_mask_branch) + 1):
                        for f in range(1, video_length):
                            noise_latents[:, :, s*video_length+f:s*video_length+f+1, :, :] = shared_noise_ratio * noise_latents[:, :, s*video_length:s*video_length+1, :, :] +\
                                (1-shared_noise_ratio) * noise_latents[:, :, s*video_length+f:s*video_length+f+1, :, :]

            noise_latents[:, :, 0:1, :, :] = first_frame_latents
            if mask_latents is not None:
                if isinstance(first_mask_latents, lst):
                    for s in range(1, np.sum(use_mask_branch) + 1):
                        noise_latents[:, :, s*video_length:s*video_length+1, :, :] = first_mask_latents[s - 1]
                else:
                    noise_latents[:, :, video_length:video_length+1, :, :] = first_mask_latents
            latents = noise_latents
        elif first_frame_cond_type == "inv":
            latents = noise_latents
        else:
            print("[Pipeline] Noise Init mode:", first_frame_cond_type)
            first_frame_latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                15,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
            if mask_latents is not None:
                if isinstance(mask_latents, list):
                    first_mask_latents = []
                    for i, mask_latent in enumerate(mask_latents):
                        first_mask_latent = self.prepare_latents(
                            batch_size * num_videos_per_prompt,
                            num_channels_latents,
                            15,
                            height,
                            width,
                            text_embeddings.dtype,
                            device,
                            generator,
                            mask_latent,
                        )
                        first_mask_latents.append(first_mask_latent)
                else:
                    first_mask_latents = self.prepare_latents(
                        batch_size * num_videos_per_prompt,
                        num_channels_latents,
                        15,
                        height,
                        width,
                        text_embeddings.dtype,
                        device,
                        generator,
                        mask_latents,
                    )
            
            #  noise_latents: 16 frames of noise
            #  first_frame_latents: 15 frames of first frame

            if shared_noise_ratio > 0:
                print("[Pipeline]Shared noise ratio =", shared_noise_ratio)
                for f in range(1, video_length):
                    noise_latents[:, :, f:f+1, :, :] = shared_noise_ratio * noise_latents[:, :, 0:1, :, :] +\
                        (1-shared_noise_ratio) * noise_latents[:, :, f:f+1, :, :]
                if mask_latents is not None:
                    for s in range(1, np.sum(use_mask_branch) + 1):
                        for f in range(1, video_length):
                            noise_latents[:, :, s*video_length+f:s*video_length+f+1, :, :] = shared_noise_ratio * noise_latents[:, :, s*video_length:s*video_length+1, :, :] +\
                                (1-shared_noise_ratio) * noise_latents[:, :, s*video_length+f:s*video_length+f+1, :, :]
            diffuse_timesteps = torch.full((1,),int(975))
            diffuse_timesteps = diffuse_timesteps.long()
            noisy_base_content = self.scheduler.add_noise(original_samples=first_frame_latents,noise=noise_latents[:,:,1:video_length,:,:],timesteps =diffuse_timesteps.to(self.unet.device))
            if mask_latents is not None:
                if isinstance(first_mask_latents, list):
                    noisy_base_content_m = []
                    for s in range(1, np.sum(use_mask_branch) + 1):
                        noisy_base_content_m.append(mask_scheduler.add_noise(original_samples=first_mask_latents[s-1],noise=noise_latents[:,:,s*video_length+1:(s+1)*video_length,:,:],timesteps =diffuse_timesteps.to(self.unet.device)))
                else:
                    noisy_base_content_m = mask_scheduler.add_noise(original_samples=first_mask_latents,noise=noise_latents[:,:,(video_length+1):(2*video_length),:,:],timesteps =diffuse_timesteps.to(self.unet.device))    

            # log (24.07): Discrete Consine Transform Initialization of noise
            if use_dct_init:
                print("[Pipeline] Noise Init mode: DCTInit")
                freq_filter = dct_low_pass_filter(dct_coefficients=first_frame_latents,
                                                            percentage=dct_cutoff_ratio, cutoff_shape=dct_cutoff_shape)
                dct_latents = exchanged_mixed_dct_freq(noise=noise_latents[:,:,1:video_length,:,:].float(),
                            base_content=noisy_base_content.float(),
                            LPF_3d=freq_filter).to(dtype=torch.float16)

                if mask_latents is not None:
                    if isinstance(first_mask_latents, list):
                        freq_filter_m = []
                        dct_latents_m = []
                        for s in range(1, np.sum(use_mask_branch) + 1):
                            freq_filter_m.append(dct_low_pass_filter(dct_coefficients=first_mask_latents[s-1],
                                                                    percentage=dct_cutoff_ratio, cutoff_shape=dct_cutoff_shape))
                            dct_latents_m.append(exchanged_mixed_dct_freq(noise=noise_latents[:,:,s*video_length+1:(s+1)*video_length,:,:].float(),
                                        base_content=noisy_base_content_m[s-1].float(),
                                        LPF_3d=freq_filter_m[s-1]).to(dtype=torch.float16))    
                    else:
                        freq_filter_m = dct_low_pass_filter(dct_coefficients=first_mask_latents,
                                                                percentage=dct_cutoff_ratio, cutoff_shape=dct_cutoff_shape)
                        dct_latents_m = exchanged_mixed_dct_freq(noise=noise_latents[:,:,video_length+1:,:,:].float(),
                                    base_content=noisy_base_content_m.float(),
                                    LPF_3d=freq_filter_m).to(dtype=torch.float16)
            else:
                print("[Pipeline] Noise Init mode: Cinemo-no DCT")
                dct_latents = noisy_base_content
                if mask_latents is not None:    
                    dct_latents_m = noisy_base_content_m

            if mask_latents is None:
                latents = torch.concat([first_frame_latents[:,:,0:1,:,:],dct_latents],dim=2)
            else:
                if isinstance(first_mask_latents, list):
                    latents = torch.concat([first_frame_latents[:,:,0:1,:,:],dct_latents],dim=2)
                    for m_latent, dct_latent_m in zip(first_mask_latents, dct_latents_m):
                        latents = torch.concat([latents, m_latent[:,:,0:1,:,:], dct_latent_m],dim=2)
                else:
                    latents = torch.concat([first_frame_latents[:,:,0:1,:,:],dct_latents,first_mask_latents[:,:,0:1,:,:],dct_latents_m],dim=2)
        # log (24.08.14): exclusive type cast for learnable text Emb setting
        if word_to_replace is not None and not is_inference:
            latents = latents.type(torch.float16)
        
        # if is_inference:
        latents = latents.to(torch.float16)
        latents_dtype = latents.dtype
        text_embeddings = text_embeddings.to(latents_dtype)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if self.attention_store is not None:
            self.attention_store.reset()

        if self.tCA_attention_store is not None:
            self.tCA_attention_store.reset()
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        last_masks = None
        last_attention_masks = {}
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                down_block_additional_residuals = mid_block_additional_residual = None
                if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                    assert controlnet_images.dim() == 5

                    controlnet_noisy_latents = latent_model_input
                    controlnet_prompt_embeds = text_embeddings

                    controlnet_images = controlnet_images.to(latents.device)

                    controlnet_cond_shape    = list(controlnet_images.shape)
                    controlnet_cond_shape[2] = video_length
                    controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)

                    controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
                    controlnet_conditioning_mask_shape[1] = 1
                    controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)

                    assert controlnet_images.shape[2] >= len(controlnet_image_index)
                    controlnet_cond[:,:,controlnet_image_index] = controlnet_images[:,:,:len(controlnet_image_index)]
                    controlnet_conditioning_mask[:,:,controlnet_image_index] = 1

                    down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                        controlnet_noisy_latents, t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_mask=controlnet_conditioning_mask,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=False, return_dict=False,
                    )

                attention_masks = {}

                if use_mask_guidance and last_masks is not None:
                    if (i % mask_inference_stride) == 0 and i <= mask_inference_cutoff:
                        masks = last_masks.clone()
                        if isinstance(use_mask_branch, bool):
                            if blurring:
                                masks = gaussian_blur(masks.float(), kernel_size=11, sigma=1.0).to(latents_dtype)
                            if "SA" in mask_guided_attention:
                                attention_masks["SA"] = interframe_mask(masks, video_length, sa_mode, device="cuda").to(latents_dtype)
                            if "CA" in mask_guided_attention:
                                attention_masks["CA"] = get_CA_mask(masks, video_length, device="cuda").to(latents_dtype) # (bf, hw) filter foreground
                            if "TA" in mask_guided_attention:
                                attention_masks["TA"] = get_TA_mask(masks, video_length, device="cuda").to(latents_dtype) # (bhw, f, f)
                            if "original" in mask_guided_attention:
                                attention_masks["SA_original"] = get_SA_original_mask(masks, video_length, device="cuda").to(latents_dtype)
                        else:
                            if "SA" in mask_guided_attention:
                                attention_masks["SA"] = []
                            if "CA" in mask_guided_attention:
                                attention_masks["CA"] = []
                            if "TA" in mask_guided_attention:
                                attention_masks["TA"] = []
                            if "original" in mask_guided_attention:
                                attention_masks["SA_original"] = get_SA_original_mask(combine_masks(masks, video_length), video_length, device="cuda", binary=True).to(latents_dtype)
                            
                            mask_idx = 0
                            for b in use_mask_branch:
                                if b:
                                    if "SA" in mask_guided_attention:
                                        attention_masks["SA"].append(interframe_mask(masks[mask_idx*video_length:(mask_idx+1)*video_length], video_length, sa_mode, device="cuda", background=False).to(latents_dtype))
                                    if "CA" in mask_guided_attention:
                                        attention_masks["CA"].append(get_CA_mask(masks[mask_idx*video_length:(mask_idx+1)*video_length], video_length, device="cuda").to(latents_dtype)) # (bf, hw) filter foreground
                                    if "TA" in mask_guided_attention:
                                        attention_masks["TA"].append(get_TA_mask(masks[mask_idx*video_length:(mask_idx+1)*video_length], video_length, device="cuda", background=False).to(latents_dtype)) # (bhw, f, f)
                                    mask_idx += 1
                                else:
                                    if "SA" in mask_guided_attention:
                                        attention_masks["SA"].append(None)
                                    if "CA" in mask_guided_attention:
                                        attention_masks["CA"].append(None)
                                    if "TA" in mask_guided_attention:
                                        attention_masks["TA"].append(None)
                        last_attention_masks = attention_masks
                    else:
                        attention_masks = last_attention_masks

                if (i % mask_inference_stride) == 0 and i <= mask_inference_cutoff:
                    # predict the noise residual
                    # print(f"{latent_model_input.dtype} {self.unet.dtype}")
                    noise_pred = self.unet(
                        latent_model_input, t, 
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals = down_block_additional_residuals,
                        mid_block_additional_residual   = mid_block_additional_residual,
                        attention_store = self.attention_store,
                        tCA_attention_store = self.tCA_attention_store,
                        attention_masks = attention_masks,
                    ).sample.to(dtype=latents_dtype)
                else:
                    noise_pred = self.unet(
                        latent_model_input[:, :, :video_length, :, :], t, 
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals = down_block_additional_residuals,
                        mid_block_additional_residual   = mid_block_additional_residual,
                        attention_store = self.attention_store,
                        tCA_attention_store = self.tCA_attention_store,
                        attention_masks = attention_masks,
                        use_mask_branch = False
                    ).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents[:, :, 1:video_length, :, :] = self.scheduler.step(noise_pred[:, :, 1:video_length, :, :], t, latents[:, :, 1:video_length, :, :], **extra_step_kwargs).prev_sample
                if mask_latents is not None and (i % mask_inference_stride) == 0 and i <= mask_inference_cutoff:
                    for s in range(1, np.sum(use_mask_branch) + 1):
                        latents[:, :, s*video_length+1:(s+1)*video_length, :, :] = mask_scheduler.step(noise_pred[:, :, s*video_length+1:(s+1)*video_length, :, :], t, latents[:, :, s*video_length+1:(s+1)*video_length, :, :], **extra_step_kwargs).prev_sample                
                
                pca_visualization = False
                initializing = False
                k = 10
                if pca_visualization and i % 5 == 0:
                    if initializing:
                        os.makedirs("temp_pca_test", exist_ok=True)
                        mask_path = ""
                        frame = latents[:, :, 0, :, :]
                        b, c, h, w = frame.shape
                        frame = rearrange(frame, "b c h w -> b h w c")
                        frame = frame.squeeze(0)
                        mask = Image.open(f"{mask_path}/4_mask.jpg").convert("L")
                        mask = mask.resize((w, h))
                        mask = torch.tensor(np.array(mask), dtype=torch.uint8)
                        location = []
                        vector = []
                        label = []

                        for i in range(h):
                            for j in range(w):
                                location.append([i, j])
                                vector.append(frame[i, j])
                                label.append(mask[i, j] > 127)

                        data = {
                            "location": location,
                            "vector": vector,
                            "label": label,
                        }
                        torch.save(data, f"temp_pca_test/t_{t}.pt")
                    else:    
                        for i in range(video_length):
                            frame = latents[:, :, i, :, :]
                            b, c, h, w = frame.shape
                            frame = rearrange(frame, "b c h w -> b h w c").squeeze(0).cpu()
                            dict = torch.load(f"temp_pca_test/t_{t}.pt", map_location="cpu")
                            res = [[[] for _ in range(w)] for _ in range(h)]
                            for idx, (x, y) in enumerate(dict["location"]):
                                d = np.linalg.norm(dict["vector"][idx] - frame[x, y])
                                res[x][y] = [d, dict["label"][idx]]
                            
                            mask = np.zeros((h, w), dtype=np.uint8)
                            for x in range(h):
                                for y in range(w):
                                    res[x][y].sort()
                                    mask[x, y] = torch.tensor(res[x][y][:k][1], dtype=torch.float32).mean() * 256
                            img = Image.fromarray(mask, mode='L')
                            os.makedirs(f"temp_pca_test/timestep_{t}", exist_ok=True)
                            img.save(f"temp_pca_test/timestep_{t}/frame_{i}.png")
                            # img.show()


                # log (250331): estimate x0 every 5 timesteps. (Usually close to final results at 50% timestep)
                if demo_x0_estimation_path is not None and i % 5 == 0:
                    from animatediff.utils.loss import estimate_x0
                    estimate_latents = latents.clone()
                    estimate_latents[:, :, 1:, :, :] = estimate_x0(latents[:, :, 1:, :, :], t, self.scheduler, noise_pred[:, :, 1:, :, :])
                    old_latents = estimate_latents.clone()
                    for f in range(1, video_length):
                        estimate_latents[:, :, f, :, :] = adaptive_instance_normalization(estimate_latents[:, :, f, :, :], old_latents[:, :, 0, :, :])
                    video = self.decode_latents(estimate_latents)
                    video = torch.from_numpy(video)
                    video = (video.clamp(0, 1).squeeze().permute(1,2,3,0) * 255).to(torch.uint8)
                    os.makedirs(f"{demo_x0_estimation_path}", exist_ok=True)
                    torchvision.io.write_video(f"{demo_x0_estimation_path}/{t}.mp4", video, fps=8)
                    os.makedirs(f"{demo_x0_estimation_path}/{t}", exist_ok=True)
                    for i, frame in enumerate(video):                          # frame: (C, H, W)
                        img = Image.fromarray(frame.numpy())
                        img.save(f"{demo_x0_estimation_path}/{t}/frame_{i}.png")

                if use_adain_norm=='on':
                    for f in range(1, video_length):
                        # original
                        old_latents = latents.clone()
                        if i > 30:
                            latents[:, :, f, :, :] = adaptive_instance_normalization(latents[:, :, f, :, :], latents[:, :, 0, :, :])
                        if f > 1:
                            latents[:, :, f, :, :] = adaptive_instance_normalization(old_latents[:, :, f, :, :], old_latents[:, :, 1, :, :])
                        # experimental - works like "last" if applied to i>30
                        # latents[:, :, f, :, :] = adaptive_instance_normalization(latents[:, :, f, :, :], latents[:, :, 0, :, :])

                if mask_latents is not None and (i % mask_inference_stride) == 0 and i <= mask_inference_cutoff:
                    from animatediff.utils.loss import estimate_x0
                    estimate_masks = latents[:, :, video_length:, :, :].clone()
                    for i in range(latents.shape[0]):
                        for s in range(1, np.sum(use_mask_branch)+1):
                            estimate_masks[i, :, (s-1)*video_length+1:s*video_length, :, :] = estimate_x0(latents[i,:,s*video_length+1:(s+1)*video_length,:,:].squeeze(0), t, mask_scheduler, noise_pred[i,:,s*video_length+1:(s+1)*video_length,:,:].squeeze(0))
                    if i == mask_inference_cutoff:
                        latents[:, :, video_length:, :, :] = estimate_masks
                    decoded_masks = []
                    for i in range(0, np.sum(use_mask_branch) * video_length):
                        decode_mask = self.decode_latents(estimate_masks[:, :, i:i+1, :, :], refl=True, normalize=True)
                        decode_mask = decode_mask.squeeze(2)
                        _, _, height, width = decode_mask.shape
                        decode_mask = F.interpolate(decode_mask, size=(height // 8, width // 8), mode='area')
                        decode_mask = decode_mask.permute(1, 0, 2, 3)
                        decoded_masks.append(decode_mask)
                        torch.cuda.empty_cache()
                    estimate_masks = torch.stack(decoded_masks, dim=1)
                    del decoded_masks
                    estimate_masks = (estimate_masks.clamp(0, 1).squeeze().permute(1,2,3,0) * 255).to(torch.uint8)

                    estimate_masks_bin = []
                    for i, frame in enumerate(estimate_masks):
                        estimate_mask_bin = rgb_to_grayscale(frame.permute(2, 0, 1))
                        estimate_masks_bin.append(estimate_mask_bin / 255)
                    estimate_masks_bin = torch.stack(estimate_masks_bin, dim=0).to("cuda")
                    last_masks = estimate_masks_bin.squeeze(1)

                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                
                if self.attention_store is not None: 
                    self.attention_store.between_steps()
                
                if self.tCA_attention_store is not None: 
                    self.tCA_attention_store.between_steps()

        # del last_masks, attention_mask, estimate_masks_bin
        # torch.cuda.empty_cache()
        # Post-processing
        if use_adain_norm=='last':              
            old_latents = latents.clone()
            for f in range(1, video_length):
                latents[:, :, f, :, :] = adaptive_instance_normalization(latents[:, :, f, :, :], old_latents[:, :, 0, :, :])
            if mask_latents is not None:
                for f in range(video_length+1, 2 * video_length):
                    latents[:, :, f, :, :] = adaptive_instance_normalization(latents[:, :, f, :, :], old_latents[:, :, video_length, :, :])
            print("[Pipeline]Applied AdaIN (mode LAST).")
        elif use_adain_norm=='on':
            print("[Pipeline]Applied AdaIN (mode ON).")
            
        old_latents = latents.clone()
        video = self.decode_latents(latents)
        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

# multi adapter masks combination
def combine_masks(masks, video_length):
    if masks.dtype is not torch.bool:
        masks = masks > 0.5
    
    masks = rearrange("(n f) h w -> n f h w", f=video_length)
    
    n, F, H, W = binary_masks.shape
    # Output: (16, H, W) — each pixel has value 0...n-1 (mask index), or n if unmasked
    # Initialize with a value higher than any valid mask index
    combined = torch.zeros((F, H, W), dtype=torch.long, device=masks.device)

    # Keep track of which pixels are already assigned
    assigned = torch.zeros((F, H, W), dtype=torch.bool, device=masks.device)

    for i in range(n):
        current_mask = binary_masks[i]  # (T, H, W)
        # Mask only unassigned pixels
        to_assign = current_mask & ~assigned
        combined[to_assign] = i + 1
        assigned |= current_mask  # update assigned mask
    
    return combined

# i2v-adapter attention mask derivation
def interframe_mask(masks, f, sa_mode, device, background=True): # only support for batch_size equals one
    h, w = masks[0].shape
    hw = h * w
    first_frame_mask = torch.zeros((f, hw, hw)).to(device)
    for i in range(f):
        first_frame_mask[i] = generate_mask(masks[i], masks[0], background)
    
    prev_frame_mask = torch.zeros((f, hw, hw)).to(device)
    for i in range(1, f):
        prev_frame_mask[i] = generate_mask(masks[i], masks[i - 1], background)
    
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
    frame_mask = torch.clamp(frame_mask, min=-20, max=0)
    return frame_mask

def generate_mask(m1, m2, binary=False, background=True):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    if binary:
        return m1[:, None] == m2[None, :]
    
    return m1[:, None] * m2[None, :] + (1 - m1[:, None]) * (1 - m2[None, :]) * background

# based-SA attention mask derivation
def get_SA_original_mask(masks, f, device, binary=False): # only support for batch_size equals one
    h, w = masks[0].shape
    hw = h * w
    attn_mask = torch.zeros((f, hw, hw)).to(device)
    for i in range(f):
        attn_mask[i] = generate_mask(masks[i], masks[i], binary)

    if binary:
        attn_mask[attn_mask == 0] = -10
        attn_mask[attn_mask == 1] = 0
    else:
        eps = 1e-4
        attn_mask = torch.log(attn_mask + eps).to(device)
        attn_mask = torch.clamp(attn_mask, min=-20, max=0)

    return attn_mask

# CA attention mask derivation
def get_CA_mask(masks, f, device):
    h, w = masks[0].shape
    # attn_mask = torch.stack(masks.clone()).to(device)
    attn_mask = masks.clone().to(device)
    # attn_mask = attn_mask > 0.5

    return attn_mask

# TA attention mask
def get_TA_mask(masks, f, device, background = True):
    h, w = masks[0].shape
    attn_mask = torch.zeros((h, w, f, f))
    # masks: shape (f, h, w)

    m = masks  # for brevity

    # Compute outer products at each (x, y): (f, h, w) → (h, w, f, f)
    term1 = (m[:, None] * m[None, :])     # shape: (f, f, h, w)
    term2 = ((1 - m[:, None]) * (1 - m[None, :]))  # shape: (f, f, h, w)

    attn_mask = term1 + term2 * background             # shape: (f, f, h, w)
    attn_mask = attn_mask.permute(2, 3, 0, 1).contiguous()  # (h, w, f, f)

    eps = 1e-4
    attn_mask = torch.log(attn_mask + eps).to(device)
    attn_mask = torch.clamp(attn_mask, min=-20, max=0)
    
    return attn_mask
