# Original version at examples/wanvideo/train_wan_t2v.py
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import random
from diffsynth.models.utils import estimate_x0, CalculateDiceLoss
from torchvision.transforms.functional import rgb_to_grayscale
import torch.nn.functional as F
from diffsynth.pipelines.wan_video import temporal_compression, interframe_mask, get_CA_mask, get_SA_full_mask, apply_guided_filter, otsu_binarization, quantile_binarization

from inference import run_inference
from lamp.data.dataset import LAMPDataset

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
EMPTY_TXT_EMB = torch.load("diffsynth/utilsempty_text_emb.pt")


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        vae_path,
        image_encoder_path=None,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        tiled=False, tile_size=(34, 34), tile_stride=(18, 16),
        use_8bit_adam=False,
        dit_kwargs:dict = {},
        use_mask_branch:bool = False,
        mask_branch_kwargs:dict = {},
        dataset = None,
        use_dice_loss = False,
        mask_only=False,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.float32, device="cpu")

        if os.path.isfile(vae_path):
            model_path = [vae_path]

        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        
        model_manager.load_models(model_path)

        dit_kwargs["use_mask_branch"] = use_mask_branch
        dit_kwargs["mask_branch_kwargs"] = mask_branch_kwargs
        # handle DiT separately here
        model_manager.load_dit(dit_path, dit_kwargs=dit_kwargs)

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.use_8bit_adam = use_8bit_adam
        self.use_mask_branch = use_mask_branch
        self.mask_branch_kwargs = mask_branch_kwargs
        self.last_masks = None
        self.sa_mode = dit_kwargs.get("SA_mode", "first")
        self.dataset = dataset
        self.timestep_ids = None
        self.use_dice_loss = use_dice_loss
        self.mask_only = mask_only
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"[train.py]{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # Data
        num_iterations_per_data = self.mask_branch_kwargs.get("num_iterations_per_data", 1)
        ground_truth_mask_probability = self.mask_branch_kwargs.get("ground_truth_mask_probability", 1)
        use_mask_guidance = self.mask_branch_kwargs.get("use_mask_guidance", False)

        if batch_idx % num_iterations_per_data == 0:
            self.timestep_ids = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (num_iterations_per_data,))
            self.timestep_ids = torch.sort(self.timestep_ids, descending=True).values

        video = batch["video"]
        mask_video = batch.get("masks_rgb", None)
        if self.mask_only:
            video = mask_video

        self.pipe.device = self.device
        if video is not None:
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            f = video.shape[2]
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0] # B,C,F,H,W
            latents = latents.unsqueeze(0).to(self.device)

            first_frame_latents = latents[:,:,:1,:,:].clone()

            prompt_emb = EMPTY_TXT_EMB
            prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
            prompt_emb["context"] = prompt_emb["context"].unsqueeze(0)
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}

        if mask_video is not None:
            mask_video = mask_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents_m = self.pipe.encode_video(mask_video, **self.tiler_kwargs)[0] # B,C,F,H,W
            latents_m = latents_m.unsqueeze(0).to(self.device)
            video_length = latents.shape[2]
            first_mask_latents = latents_m[:,:,:1,:,:].clone()
            latents = torch.concat([latents, latents_m], dim=2)

        attention_masks = {}
        if use_mask_guidance:
            ground_truth_mask_probability = self.mask_branch_kwargs.get("ground_truth_mask_probability", 1.0)
            mask_guided_attention = self.mask_branch_kwargs.get("mask_guided_attention", "SA")
            binarization_mode = self.mask_branch_kwargs.get("binarization_mode", None)
            guided_filter = self.mask_branch_kwargs.get("guided_filter", False)
            if random.random() < ground_truth_mask_probability or self.last_masks is None:
                masks = batch["masks_bin"].squeeze(0)
            else:
                masks = self.last_masks
            # if blurring:
            #     masks = gaussian_blur(masks.float(), kernel_size=11, sigma=1.0).to(weight_dtype)
            masks = temporal_compression(masks)
            if guided_filter:
                masks = apply_guided_filter(masks)
            if binarization_mode == "otsu":
                masks = otsu_binarization(masks)
            elif binarization_mode == "quantile":
                masks = quantile_binarization(masks)

            if "SA" in mask_guided_attention:
                attention_masks["SA"] = interframe_mask(masks, video_length, self.sa_mode, device="cuda").to(self.pipe.torch_dtype)
            if "CA" in mask_guided_attention:
                attention_masks["CA"] = get_CA_mask(masks, video_length, device="cuda").to(self.pipe.torch_dtype) # (b, fhw) filter foreground
            if "full" in mask_guided_attention:
                attention_masks["full"] = get_SA_full_mask(masks, video_length, device="cuda").to(self.pipe.torch_dtype)

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss
        self.pipe.device = self.device

        timestep_id = self.timestep_ids[batch_idx % num_iterations_per_data].unsqueeze(0)
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noise = torch.randn_like(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        # Replace noise
        noisy_latents[:,:,:1,:,:] = first_frame_latents
        if self.use_mask_branch:
            noisy_latents[:, :, video_length:video_length+1, :, :] = first_mask_latents

        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            attention_masks=attention_masks
        )

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # if self.use_mask_branch:
        #     loss = torch.nn.functional.mse_loss(noise_pred[:, :, :video_length, :, :].float(), training_target[:, :, :video_length, :, :].float())

        if self.use_mask_branch:
            with torch.no_grad():
                estimate_masks = noisy_latents[:, :, video_length:, :, :].clone()
                for i in range(noisy_latents.shape[0]):
                    estimate_masks[i, :, 1:, :, :] = estimate_x0(noisy_latents[i,:,video_length+1:,:,:].squeeze(0), timestep_id, self.pipe.scheduler, noise_pred[i,:,video_length+1:,:,:].squeeze(0))
                decoded_masks = self.pipe.decode_video(estimate_masks)
                b, c, _, height, width = decoded_masks.shape
                decoded_masks = decoded_masks.reshape(-1, 1, height, width)
                decoded_masks = F.interpolate(decoded_masks, size=(height//8, width//8), mode='area')
                decoded_masks = decoded_masks.reshape(c, f, height//8, width//8)
                estimate_masks = decoded_masks
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
            self.last_masks = estimate_masks_bin.squeeze(1)

            if self.use_dice_loss and use_mask_guidance:
                dice_loss = CalculateDiceLoss(estimate_masks_bin > 0.5, batch["masks_bin"].squeeze(0).unsqueeze(1))
                print(f"L2 loss: {loss}, DICE loss: {dice_loss}")
                loss = torch.add(loss, dice_loss.mean())

        loss = loss * self.pipe.scheduler.training_weight(timestep)
        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(trainable_modules, lr=self.learning_rate)
        print("[LightningModelForTrain.configure_optimizers]Optimizer:", optimizer)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


class CustomModelCheckpoint(pl.pytorch.callbacks.Callback):
    def __init__(self, config, validation_pipeline, save_freq=20, val_freq=20, save_every_n_epochs=5):
        super().__init__()
        self.config = config # This should be the OmegaConf object
        self.save_freq = save_freq
        self.val_freq = val_freq
        self.save_every_n_epochs = save_every_n_epochs
        self.steps_done = 0
        self.pipe = validation_pipeline # the Pipeline instance for validation

    def _run_validation(self, trainer, label):
        if trainer.logger and trainer.logger.log_dir:
            base_output_dir = os.path.dirname(trainer.logger.log_dir)
        else:
            base_output_dir = trainer.default_root_dir
            if not os.path.exists(base_output_dir):
                 os.makedirs(base_output_dir, exist_ok=True)

        val_output_dir = os.path.join(base_output_dir, "samples", label)
        os.makedirs(val_output_dir, exist_ok=True)

        print(f"[train.py]Running validation @ {label}")

        run_inference(
            output_dir=val_output_dir,
            validation_data=self.config.validation_data,
            validation=self.config.validation,
            num_persistent_param_in_dit=self.config.get("num_persistent_param_in_dit", None), # not sure of how this works
            pipe=self.pipe,
            empty_txt_emb=True,
            use_mask_branch=self.config.get("use_mask_branch", False),
            mask_branch_kwargs=self.config.get("mask_branch_kwargs", {}),
            mask_only=self.config.get("mask_only", False)
        )

        self.pipe.scheduler.set_timesteps(1000, training=True)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.steps_done += 1
        if self.steps_done % self.save_freq == 0:
            ckpt_dir = os.path.join(trainer.logger.log_dir if trainer.logger and trainer.logger.log_dir else trainer.default_root_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{self.steps_done}.ckpt")

            trainer.save_checkpoint(ckpt_path)

        if self.steps_done % self.val_freq == 0:
            self._run_validation(trainer, f"step_{self.steps_done}")

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.dataset.shuffle()


def train(config):
    dataset = LAMPDataset(
        video_root=config.dataset_path,
        mask_root=config.get("mask_path", None),
        prompt=config.prompt,
        n_sample_frames=config.num_frames,
        height=config.height,
        width=config.width,
        sample_start_idx=0,
        sample_frame_rate=config.get("sample_frame_rate", 1),
        num_iterations_per_data=config.get("mask_branch_kwargs", {}).get("num_iterations_per_data", 1)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,     ## random shuffle the examples in training dataset
        batch_size=1,
    )
    model = LightningModelForTrain(
        dit_path=config.dit_path,
        vae_path=config.vae_path,
        learning_rate=config.learning_rate,
        train_architecture=config.train_architecture,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_target_modules=config.lora_target_modules,
        init_lora_weights=config.init_lora_weights,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=config.use_gradient_checkpointing_offload,
        pretrained_lora_path=config.pretrained_lora_path,
        use_8bit_adam=config.use_8bit_adam,
        dit_kwargs=config.dit_kwargs,
        use_mask_branch=config.get("use_mask_branch", False),
        mask_branch_kwargs=config.get("mask_branch_kwargs", {}),
        dataset=dataset,
        use_dice_loss = config.get("use_dice_loss", False),
        mask_only = config.get("mask_only", False)
    )

    # set (and print) trainable parameters
    print_trainable_params = config.get("print_trainable_params", False)
    if print_trainable_params:
        os.makedirs(config.output_path, exist_ok=True)
        path_params = os.path.join(config.output_path, "trainable_params.txt")
        lines = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        for module in config.trainable_modules:
            if module in name:
                param.requires_grad = True
                if print_trainable_params:
                    lines.append(name)
    if print_trainable_params:
        with open(path_params, 'w') as f:
            f.write('\n'.join(lines))
        print("[train.py]Trainable params written to:", path_params)
    
    if config.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        # Use OmegaConf.to_container to convert OmegaConf object to dict for SwanLab
        swanlab_config.update(OmegaConf.to_container(config, resolve=True)) 
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=config.swanlab_mode,
            logdir=os.path.join(config.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None

    custom_checkpoint_callback = CustomModelCheckpoint(config,
                                                       save_freq=config.checkpointing_steps,
                                                       val_freq=config.validation_steps,
                                                       validation_pipeline=model.pipe,
                                                       )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="32-true",
        strategy=config.training_strategy,
        default_root_dir=config.output_path,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[custom_checkpoint_callback,],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    train(config)