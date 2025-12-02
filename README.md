# MIVA: Few-Shot-Based Modular Image-to-Video Adapter for Diffusion Models

This repository exhibits the core algorithm described in the paper.

Our codes are based on the following works. Many thanks to the authors.
* Diffusers: https://github.com/huggingface/diffusers
* AnimateDiff: https://github.com/guoyww/AnimateDiff
* LAMP: https://github.com/RQ-Wu/LAMP
* DiffSynth: https://github.com/modelscope/DiffSynth-Studio
* Wan Video: https://github.com/Wan-Video/Wan2.1

## AnimateDiff version

Dependent models & datasets
* Stable Diffusion v1.5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
* Motion module: https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt
* LAMP dataset: https://github.com/RQ-Wu/LAMP

Core components
* Architecture (including parallelism)
  * SA adapter (CFA layers): `animatediff/models/sa_utils.py`
  * CA adapter: `animatediff/models/ca_utils.py`
  * s-TA adapter: `animatediff/models/motion_module.py`
* Inference-time techniques: `animatediff/pipelines/pipeline_animation.py`

Both training and inference rely on YAML configs (examples at `animatediff/configs`).

## Wan2.1 version

Dependent models & datasets
* Wan2.1: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
  * VAE: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/Wan2.1_VAE.pth
* LAMP dataset: https://github.com/RQ-Wu/LAMP

The core components are mostly migrated from AnimateDiff version, with light modifications for compatibility.