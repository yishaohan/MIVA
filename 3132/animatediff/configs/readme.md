- [Config README](#config-readme)
  - [Example](#example)
  - [Header - Base checkpoints](#header---base-checkpoints)
  - [Header - Load the adapters](#header---load-the-adapters)
    - [`ckpt_mode`](#ckpt_mode)
    - [Training continuing from previous ckpt](#training-continuing-from-previous-ckpt)
    - [Use multiple adapters](#use-multiple-adapters)
  - [Header - Learning textual embeddings](#header---learning-textual-embeddings)
  - [Header - Other values for training](#header---other-values-for-training)
    - [`optical_flow_kwargs`](#optical_flow_kwargs)
    - [`perceptual_loss_kwargs`](#perceptual_loss_kwargs)
    - [`dino_loss_kwargs`](#dino_loss_kwargs)
  - [Section `unet_additional_kwargs`: the UNet architecture](#section-unet_additional_kwargs-the-unet-architecture)
    - [SA](#sa)
    - [Motion module (TA)](#motion-module-ta)
      - [`motion_module_kwargs`](#motion_module_kwargs)
    - [CA-LoRA](#ca-lora)
    - [Spatiotemporal $Q$ in CA](#spatiotemporal-q-in-ca)
    - [Simple CA](#simple-ca)
    - [`simple_CA_kwargs`](#simple_ca_kwargs)
    - [Mask branch](#mask-branch)
    - [`mask_branch_kwargs`](#mask_branch_kwargs)
    - [Temporal CA](#temporal-ca)
    - [Orthogonal Adapters (OA)](#orthogonal-adapters-oa)
    - [Multi-adapter setting](#multi-adapter-setting)
    - [Zero-out mechanism](#zero-out-mechanism)
  - [Section `train_data`](#section-train_data)
  - [Section `validation_data`](#section-validation_data)
    - [Input](#input)
    - [Output](#output)
    - [Sampling process](#sampling-process)
    - [Initial noise](#initial-noise)
    - [Multi-adapter](#multi-adapter)
    - [Learnable text embedding](#learnable-text-embedding)
    - [Visualizing CA maps](#visualizing-ca-maps)
  - [Section `noise_scheduler_kwargs`](#section-noise_scheduler_kwargs)
  - [Training \& inference under inversion setting (experimental)](#training--inference-under-inversion-setting-experimental)
    - [Root level](#root-level)
    - [Data](#data)
    - [Training](#training)
    - [Validation \& inference](#validation--inference)


# Config README

The config file structure is largely based on AnimateDiff, but we make some differences.

## Example
```YAML
# Header
output_dir: "outputs/inference_adapter"
pretrained_model_path: "/data/diffusion/hf/stable-diffusion-v1-5"
motion_module_path: "/data/diffusion/SD_AnimateDiff/models/Motion_Module/v3_sd15_mm.ckpt"
use_ip_adapter: False

# ===== CKPT PATH: CHOOSE ONE OF THE MODE BELOW =====
# In inference, ckpt path can be a list. In training, ckpt path should always be a str.
# Option 1: load multiple full ckpts; each ckpt contains a full UNet including frozen weights
ckpt_mode: full
unet_checkpoint_path: "temp_ckpts/task1/checkpoints_2000/model.safetensors"

# Option 2: load fine-tuned weights only (separate file)
ckpt_mode: split
adapter_i2v_path : temp_ckpts/task2/checkpoints_2000/animatediff-i2v-nv1-motionlora-i2v-weights.pt # inference: can be multiple; training: only one
adapter_lora_path : temp_ckpts/task2/checkpoints_2000/animatediff-i2v-nv1-motionlora-lora-weights.pt
ca_lora_path : temp_ckpts/task2/checkpoints_2000/animatediff-i2v-nv1-ca-lora-weights.pt

## Option 3 (new): load fine-tuned weights only (single adapter file)
ckpt_mode: adapter
unet_checkpoint_path : temp_ckpts/task3/checkpoints_2000/adapter.pt
 
# Training loss settings (for training configs)
include_frame1_in_mse_loss: False

unet_additional_kwargs:
  use_motion_module              : true
  motion_module_resolutions      : [ 1,2,4,8 ]
  unet_use_cross_frame_attention : false
  unet_use_temporal_attention    : false

  motion_module_type: Vanilla
  motion_module_kwargs:
    num_attention_heads                : 8
    num_transformer_block              : 1
    attention_block_types              : [ "Temporal_Self", "Temporal_Self" ]
    temporal_position_encoding         : true
    temporal_position_encoding_max_len : 32
    temporal_attention_dim_div         : 1
    zero_initialize                    : true
    lora_rank: 32
    lora_scale: 1
  
  adapter_weights : [0.3, 0.7]

noise_scheduler_kwargs:
  num_train_timesteps: 1000
  beta_start:          0.00085
  beta_end:            0.012
  beta_schedule:       "scaled_linear"
  steps_offset:        1
  clip_sample:         false


validation_data:
  image_path: "benchmark/waterfall"
  prompt_path:
    - "lava_waterfall"
    - "waterfall_and_a_Ferrari"
  prompts:
    - "waterfall"
    - "waterfall"
  video_length: 16
  width: 512
  height: 320
  num_inference_steps: 50
  guidance_scale: 1
  use_inv_latent: False
  num_inv_steps: 50
```

## Header - Base checkpoints
The first part of the config file specifies the skeleton of the video DM.
* Essentially it is a T2I model (e.g. SD) + motion modules aka temporal AttBlocks (e.g. AnimateDiff) + optionally IP-Adapter.
  
| Name        | Notes           |
| ------------- |-------------| 
| `output_dir`    | Where the inference results will be stored |
| `pretrained_model_path`     | Path to the pretrained T2I model   |
| `motion_module_path` | Path to the pretrained motion module (AnimateDiff) |
| `use_ip_adapter`: "IP-Adapter" / "ID-Animator" / `None` | Whether to use IP-Adapter, ID-Animator or not (choose from `IP-Adapter` / `ID-Animator` / `None`). Default = `None`. |
| `ip_ckpt`     | (if `use_ip_adapter==True`) Path to the pretrained IP-Adapter   |
| `image_encoder_path` | (if `use_ip_adapter==True`) Path to the pretrained image encoder (in IPAdapter) |
| `repeat_token` | Optional. Whether to repeat token in text embedding. Has three modes: full (77 word tokens), except_bos (bos + 76 word token), noise (word token + 76 * (word token + noise)), noise_after (noise applied after embeddings passed through CLIP encoder). |
| `textemb_checkpoint_path` | Optional. The path to the textual embedding for [customized textual embedding case](#learnable-text-embedding). |
| `global_seed`: int | The global seed for random number generation. |
| `ckpt_mode`: "full" / "split" / "adapter" | Ckpt saving mode. `full`: save the entire model, including the pre-trained weights. `split`: save trainable weights only, separated into 3 files (SA, CA and TA respectively). `adapter`: save trainable weights only as a single adapter file. Default = `full`. |
| `pretrained_sa_adaptive_weight_module` | Optional. If given, the model loads the weights of SA adaptive weighting modules from an individual file, overriding the weights from the ckpt. Default = None. |

## Header - Load the adapters 

### `ckpt_mode`
* `full` (default) \
Traditionally, all weights including those not fine-tuned are saved into one `.safetensors` file per training task. In this case, users need to specify
  * `unet_checkpoint_path`
* `split` \
Save/load fine-tuned weights only, in which case a task produces at most 3 checkpoints. In this case, users need to specify:
  * `adapter_i2v_path`: The I2V-Adapter weights at SA.
  * `adapter_lora_path`: The MotionLoRA at temporal layers.
  * `ca_lora_path`: The CA-LoRA at CA.
  * Any attribute can be left as `None` if not involved in the training task.
* `adapter` \
Save/load fine-tuned weights only, but all parameters are packed into a single `.pt` ckpt file. In this case, users need to specify:
  * `unet_checkpoint_path`

### Training continuing from previous ckpt

| Name        | Notes           |
| ------------- |-------------| 
|`unet_checkpoint_path`: str| Set to path of the previous ckpt wishing to continue training from. Leave blank if training from scratch. |
|`unet_checkpoint_starting_step`: int| Specify which step to start training from. Supposed to set manually, and must match the number of steps of the ckpt specified in `unet_checkpoint_path`.|

### Use multiple adapters

NOTE: inference only - not supported by training.

In this case, all the above attributes should be a list. Example:

```yaml
# Option 1: load multiple full ckpts; each ckpt contains a full UNet including frozen weights
load_from_adapter_ckpt : False
unet_checkpoint_path: 
  - "checkpoints/model1.safetensors"
  - "checkpoints/model2.safetensors"

# Option 2: load fine-tuned weights only
# NOTE: all list lengths should match.
load_from_adapter_ckpt : True
adapter_i2v_path : 
  - adapters/Adapter1-i2v-weights.pt # inference: can be multiple; training: only one
  - adapters/Adapter2-i2v-weights.pt
adapter_lora_path : 
  - adapters/Adapter1-lora-weights.pt
  - adapters/Adapter2-lora-weights.pt
ca_lora_path : 
  - adapters/Adapter1-ca-lora-weights.pt
  - adapters/Adapter2-ca-lora-weights.pt
```

## Header - Learning textual embeddings

Below arguments work in training only.

| Name        | Notes           |
| ------------- |-------------| 
|`word_to_learn`: str| If set, the training task also learn a new word embedding for the specified word. Don't include this if the task won't learn word embeddings. |
|`initializer_token`: str| If `word_to_learn` is set, the trainer initializes the word embedding of `word_to_learn` as this. |

* Example: `word_to_learn` = "clouds_new", `initializer_token` = "clouds"

Sample Prompts for all cases (by Ambrose):
* waterfall: `waterfall,flowing,nature,cascading,waterfall_new`
* fireworks: `firework,colourful sparks,bursting,flaring,firework_new`
* guitar: `playing the guitar, strumming, plucking strings, guitar new`
* smile: `turn to smile,happy,excited,smile_new`
* birdsfly: `birds flying, soaring, flapping wings, birds_fly_new`
* helicopter: `helicopter, blades propelling, spinning, helicopter_new`
* rain: `raining, raindrops, water droplets falling, dripping, rain_new`

## Header - Other values for training

Below arguments work in training only.

| Name        | Notes           |
| ------------- |-------------| 
| `split_ckpt`: bool | `True` to store only the fine-tuned weights, `False` to store all UNet weights in a `.safetensors` file. Default = `False`. |
|`include_frame1_in_mse_loss`: bool | If True, the L2 loss will include the first frame too. Default = `False`. |
|`use_optical_flow`: bool | `True` to switch on optical flow (OF) features, e.g., the losses. |
|`optical_flow_kwargs`: dict | See below. |
|`use_perceptual_loss`: bool | `True` to switch on perceptual loss features.|
|`perceptual_loss_kwargs`: dict | See below. |
|`use_dino_loss`: bool | `True` to switch on DINO loss features. |
|`dino_loss_kwargs`: dict | See below. |
|`log_losses_in_csv`: bool | `True` to store the value of losses in a csv file. The system logs each loss term every 100 iterations. Default = `False`. |
|`print_trainable_params`: bool| `True` to store the list of trainable parameters as "trainable_params.txt" under `output_dir`. Default = `False`.|
|`use_dice_loss`: bool| `True` to switch on DICE loss features. For each training iteration, apply DICE loss on one random mask of the mask sequence. Default = `False`.|

### `optical_flow_kwargs`

| Name        | Notes           |
| ------------- |-------------|
|`use_refl`: bool | If True, perform ReFL to estimate x0 from x_t at current timestep t. Default = `False`.|
|`weight`: float | As is. Default = 1.|
|`raft_path`: str | Path to the RAFT (OF model) ckpt.|
|`frame_extraction_mode`: 'random' / 'consecutive' / 'causal_random' | In "consecutive" mode, the batch is formed by extracting frames sequentially, such as [frame1, frame2], [frame2, frame3], … [frame14, frame15]. In "random" mode, the batch is formed by selecting random frames from the video. For example, [frame3, frame1], [frame5, frame9], …. In "causal_random" mode, the batch formation is the same as in "random" mode, but the second frame index is always greater than the first frame index. Default = `random`.|
|`use_cosine_schedule`: bool | This flag determines whether to apply a cosine scheduling factor to the warping loss function based on the timestep. The cosine schedule is a monotonically decreasing function from 1 to 0, meaning that for noisier timesteps, the warping loss is reduced more. Default = `False`.|
|`use_tv_loss`: bool | This flag determines whether total variation regularization should be applied to the optical flow. Enabling it will produce smoother motion but may affect the dynamics. Default = `False`.|
|`weight_tv`: float | The coefficient of the TV loss function.  Default = 0.05.|
|`use_cosine_schedule_tv`: bool | Same as use_cosine_schedule_for_warping_loss for TV regularization loss.  Default = `False`.|
|`visualize`: bool | This flag saves the visualized results for pairs of input frames used in optical flow estimation, along with the warped version of the frame using the predicted optical flow. It also visualizes the optical flow itself and a boundary mask, which defines the valid regions for optical flow. Optical flow values that fall outside the image will be set to zero. Default = `False`.|

### `perceptual_loss_kwargs`

Note that in the current implementation the loss is always computed from estimated $x_0$ at step time $t$.

| Name        | Notes           |
| ------------- |-------------| 
|`vgg_path`: str | Path to the VGG16 model. |
|`weight`: float | As is. Default = 1.|
|`cutoff`: int | If set, then only apply the loss when time step <= `cutoff`.|
|`use_cosine_schedule`: bool | Same as use_cosine_schedule_for_warping_loss for perceptual loss.  Default = `False`.|
|`reference_frame`: str |This parameter can take the value "first" or "current". If set to "first", the ground truth features used in the perceptual loss are extracted from the first frame. Otherwise, if set to "current", the ground truth features are taken from the current frame. Default = `current`.|
|`visualize`: bool | If `True`, save the comparison between a random predicted frame and the ground truth every 200 iterations. |

### `dino_loss_kwargs`

Note that in the current implementation the loss is always computed from estimated $x_0$ at step time $t$.

| Name        | Notes           |
| ------------- |-------------| 
|`dino_path`: str | Path to the DINO model. (Default is a DINO-v1-ViT-B/16) |
|`weight`: float | As is. Default = 1. |
|`use_cosine_schedule`: bool | Same as use_cosine_schedule_for_warping_loss for perceptual loss.  Default = `False`.|
|`reference_frame`: str |This parameter can take the value "first" or "current". If set to "first", the ground truth features used in the dino loss are extracted from the first frame. Otherwise, if set to "current", the ground truth features are taken from the current frame. Default = `current`.|
|`visualize`: bool | If `True`, save the comparison between a random predicted frame and the ground truth every 200 iterations. |


## Section `unet_additional_kwargs`: the UNet architecture

This section specifies the DM's architecture.
The loaded checkpoints must follow the definition here or misalignment errors may occur.

NOTE: some attributes are omitted - please just don't modify them.

### SA

| Name        | Notes           |
| ------------- |-------------| 
|`sa_mode`: "first" (default) / "first_prev" / "prev" | Which frame to use to condition SA. |
|`sa_first_frame_scale`: float / str | Works when sa_mode = "first_prev" ONLY. Determine the ratio between the weight of first frame and prev frame, or the weight function. Default = 0.5. If `linear`, then the weight shifts from 100% first frame at the last timestep and 0% previous frame at timestep 0. If `adaptive_TTT`, then each I2V-Adapter layer contains a learnable module, named "weight_module", to decide the weight given timestep. TTT is the associated temperature (float, default = 1).|

### Motion module (TA)

| Name        | Notes           |
| ------------- |-------------| 
|`use_motion_module`: bool    | Whether to use motion modules (temporal AttBlocks) |
|`use_motion_embed`: bool | Under `motion_module_kwargs` section. If True, then learn a motion embedding following [Motion-Inv]. |

#### `motion_module_kwargs`

| Name        | Notes           |
| ------------- |-------------| 
|`lora_rank`: int / null | Apply LoRA to TA. |
|`adapter_weights`: list[float] | Refer to [Multi-adapter setting](#multi-adapter-setting).|
|`parallel_mode`: "weights" / "residual" | Refer to [Multi-adapter setting](#multi-adapter-setting).|
|`use_simple_CA`: bool | If `True`, use simple CA layer after each TA layer. Default = `False`.|
|`kv_emb_len`: int | If >0, learn K and V embedding per layer with token length = `kv_emb_len` each. Default = 0. |

`simple_CA_kwargs`
* `skip_w_q`: default = `False`. If `True`, skip $W_Q$ ($CA_1$) matrix.
* `rank`: default = 64.
* `bias`: default = `False`.
* `use_pretrained_w_q`: if `True` (default), $CA_1 \equiv W_Q$ from the TA layer, not trainable.
* `use_pretrained_w_out`: if `True`, have a $CA_4 \equiv W_O$ from the TA layer. Default = `False`.
* `nonlinear`: if `True` (default), add nonlinear operations (LN+GM) between $CA_2, CA_3$.


### CA-LoRA

| Name        | Notes           |
| ------------- |-------------| 
|`use_ca_lora`: bool    | Whether to use CA-LoRA |
|`ca_lora_rank`: int / list[int]    | Rank of CA-LoRA(s). For single adapter, it's an *int* (e.g. `32`); for multi-adapter, it's a list of ranks for each adapter (e.g. `[32,32]`)|
|`ca_lora_scale`: float / list[float]    | `network_alpha` value of CA-LoRA(s). See lora.py for details. Data type works the same as `ca_lora_rank`|
|`use_q_lora`: bool  | if True, attach LoRA to W_Q. Default = `False` (only attach LoRAs to W_K,V,O).|

### Spatiotemporal $Q$ in CA

| Name        | Notes           |
| ------------- |-------------| 
|`q_downsample`: bool    | Whether to use spatiotemporal $Q$ in CA. |
|`q_downsample_ratio`: int    | The stride of spatial downsampling in $Q$. Default = 4.|
|`ca_pe_mode`: None ("null") / "naive" / "temporal" / "temporal_sine" / "rope_1d" / "rope_3d" or "ropeQ_3d" / "ropeQK_1d" / "ropeQKV_1d" / "ropeQK_3d" / "ropeK_3d" / "ropeQKV_3d" | The positional embedding setting for the query tensor in CA. Default = None. |
|`use_CA_att_mask_framewise`: bool | Whether to use frame_filter during CA. frame_filter_attention_mask is applied during CA attention calculation, and filters pixel-patches to corresponding tokens that belongs to the same frame.|

### Simple CA

Simple CA replaces the CA layers by an unconditional MLP: $f(x)= \phi (xW_Q W_1^T) W_2$.
* $W_1, W_2 \in \mathbb{R}^{r \times d}$. $d$ = output channel # of $W_Q$; $r$ = rank.
* $\phi$ is an activation (We use GeLU here).

| Name        | Notes           |
| ------------- |-------------| 
|`use_simple_CA`: bool | Whether to use simple CA (code at attention.py). Default = `False`.|

### `simple_CA_kwargs`
| Name        | Notes           |
| ------------- |-------------|
|`rank`: int| Rank of the CA matrices. Default = 64. |
|`norm`: bool| If true, will use normalization before smplCA. Default = False |
|`out`: bool| If true, will use W_out (CA4) after smplCA. Default = `False` |
|`norm_mid`: string| Specify what normaliztion to use between CA2 and CA3. Default = `layernorm` |
|`act`: string| Specify what activation to use between CA2 and CA3. Default = `gelu` |


### Mask branch

| Name        | Notes           |
| ------------- |-------------| 
|`use_mask_branch`: bool / list[bool] | Whether to use mask branch of MIVA. Default = `False`. Provide a list of bool when using multiple MIVA adapters|
|`mask_inference_stride`: int | Inference only: Specify the length of stride used when determining which inference step would denoise mask sequence. For instance, a value of 5 would means the mask sequence would be denoised every 1 out of 5 steps. Default: 1 |
|`mask_inference_cutoff`: int | Inference only: Specify at which step mask sequence would no longer be denoised (inclusive). For instance, a value of 35 indicates mask sequence would no longer be denoised at and after inference step 35. Default: 50|

### `mask_branch_kwargs`
| Name        | Notes           |
| ------------- |-------------|
|`use_mask_guidance`: bool | Whether to use mask sequence to generate attention mask. Default = `False` but should always be `True` when using mask branch. (Henry: results would be disastrous if `False`) |
|`use_image_guidance`: bool| If `True`, use $I^i$ to guide generation of $S^i$ when generating the mask sequence. Default = `False` |
|`mask_guided_attention`: `SA` / `CA` / `TA` / `original` / any combination of above module(s) with '_' | Specifiy which attention layer will be guided using mask sequence derived attention mask. Default = `` |
|`blurring`: bool| If `True`, use Gaussian blur on mask sequence before deriving attention mask. Default = `False` |
|`num_iterations_per_data`: int | Training only: Specify how many times a sampled video segment is used during training. Default = 1 |
|`ground_truth_mask_probability`: float / 'cosine' | Training only: dropout probabilty. Specify what probability the ground truth mask sequence is used to derive attention mask. Default = 1 |


### Temporal CA

| Name        | Notes           |
| ------------- |-------------| 
| `use_temporal_CA`: bool | Whether to use temporal CA layers. They are now placed between vanilla CA and TA. |

### Orthogonal Adapters (OA)

Leave this part blank if not used (remove the related attributes at all).

| Name        | Notes           |
| ------------- |-------------| 
|`oa_bin_id`: int    | The bases bin ID. An adapter w.r.t. a particular movement should have a unique ID. |
|`oa_bases_path`: str    | The path that stores the pre-computed LoRA bases. Users are assumed to run `scripts/lora_basis_lib/create.py` to generate the bases first.|

### Multi-adapter setting
| Name        | Notes           |
| ------------- |-------------| 
| `adapter_weights`: list[float]   | FOR INFERENCE ONLY. Weight of each adapter during inference. Default = `[1]`. For two-adapter setting, `[0.5,0.5]` can be a good start.|
| `parallel_mode`: "weights" / "residual" | FOR INFERENCE ONLY. Determines how multiple adapters are parallelized. Default = "residual". "weights" follows original LoRA practice by adding all LoRAs to the pre-trained weights.|

### Zero-out mechanism
| Name        | Notes           |
| ------------- |-------------| 
| `zero_out_first_frame`: bool  |  Zero out the first frame values of all residual tensors. Default = `False`.|

## Section `train_data`

The setting for the dataset instance during training.
Refer to `lamp/data/dataset.py` for details.

| Name        | Notes           |
| ------------- |-------------| 
| `video_root`: str   | Dataset path. Can be a json file or a folder containing all training videos.|
| `mask_root`: str   | Mask images path. should correspond to the training videos given by `video_root`. |
| `prompt`: str | The prompt paired with each video (In our setting, all videos are paired with this prompt, which may not hold for other video generation tasks) |
|`n_sample_frames`: int| Video length. Should be consistent with the video length in validation/inference. Default = 16.|
|`width`, `height`: int| The dataset instance will resize every video to this resolution. Default = 512, 320. |
|`sample_start_index`: int| Deprecated.|
|`sample_frame_rate`: int / list[int, int]| The sampling rate with which the system samples frames from the raw video to form the training frame sequence. If `int`, the sampling rate is fixed for all videos. If `list` (specifying an interval), then the dataset will randomly draw a number from the interval at each iteration. The system will automatically avoid too large sampling rate (in which case, the video is too short to cover the target sequence). |
|`aug`: str   | Data augmentation. Can be any combination of "flip", "crop" and "color". Leave this blank if not performing any augmentation. Default = "flip".|


## Section `validation_data`

The setting for validation phase (training) and inference (inference). 

### Input

| Name        | Notes           |
| ------------- |-------------| 
|`image_path`: str| The path to the input images (assuming all input images are stored in the same folder)|
|`mask_path`: str| The path to the input mask images (assuming all mask images are stored in the same folder)|
|`prompt_path`: list[str] | List of image file names. The program will run through all files. |
|`prompts`: list[str] / "all" / "newlamp"| A list whose length == `prompt_path`, specifying the text prompt per input image. |
|`batch_prompt`: None(default) / str / list[str]| Use this if the prompt is consistent for all input images. *In multi-adapter case,* `batch_prompt` will override `prompts` and must be a list of str, specifying the prompt injected to each adapter respectively (e.g. `["waterfall", "cloud"]`). |

### Output
| Name        | Notes           |
| ------------- |-------------|
|`video_length`: int| As is. Default = 16.|
|`width`, `height`: int| As is. Default = 512, 320.|
|`demo_x0_estimation_path`: str: If given, will perform 1 step estimation and store result at 5, 10, 15, ... 50 steps. Results are stored at given path. Default = `` |

### Sampling process
| Name        | Notes           |
| ------------- |-------------|
|`num_inference_steps`: int| Time step # for the sampler. Default = 50.|
|`guidance_scale`: float | CFG guidance scale|
|`use_adain_norm`: "off" / "on" / "last" | AdaIN behaviour in the sampling process. "off": no AdaIN. "on": perform AdaIN after each timestep (by [LAMP]); perform AdaIN only after the last timestep. Default = "off".|

### Initial noise
| Name        | Notes           |
| ------------- |-------------|
|`shared_noise_ratio`: float | Ratio of shared noise across all frames in $z_T$. [LAMP] uses 0.2. We find 0.05 is good for our setting. Default = 0. |
|`first_frame_cond_type`: "lamp" / "cinemo" | "lamp": Initial noise is pure noise with optional shared-noise mechanism. "cinemo": Initial noise is a noisy signal by forward diffusion process. Must be "cinemo" if the user would use DCTInit initalization. Default = "lamp". |
|`use_dct_init`: bool| If `True` and "cinemo" flag, apply DCTInit - integrate low-frequency component of the latent image and high-frequency component of random noise as the initial noise. |
|`dct_cutoff_ratio`: float| (for DCTInit case only) The cutoff ratio in [0,1] for DCTInit. Default = 0.23. |
|`dct_cutoff_shape`: 'rect' / 'tri'| (for DCTInit case only) How DCT frequency domain is divided at top left, in rectangle or triangle shape. default = "rect". |
|`cil_ratio`: float| Default = 1. Otherwise, all the timesteps in the schedule will be multiplied by it following [CIL]. |

### Multi-adapter
| Name        | Notes           |
| ------------- |-------------|
|`multi_ca`: bool| Set as `True` to switch on multi-adapter setting. Default = `False`. |
|`batch_prompt`: list[str]| See above.|
|`text_emb_dir`: str| (for multi-adapter case only) If given, the program will obtain the text embeddings offline. Specifically, collect all prompts from `batch_prompt`, and then search for corresponding embedding file in this folder; for example, if a prompt is "waterfall", then the code will try to load `text_emb_dir/waterfall.pt`. Users must generate the embedding in advance to enable this feature. Default = "text_embs". |

### Learnable text embedding

| Name        | Notes           |
| ------------- |-------------|
|`word_to_replace`: str / list[str] | If set, the program will replace the embedding of this word with a learned embedding. Should be a list under multi-adapter setting. |

* The embedding is specified by `textemb_checkpoint_path` in the header. Otherwise the code assumes
  * UNet ckpt at `unet_checkpoint_path`/checkpoints/model.safetensors
  * Text-emb at `unet_checkpoint_path`/learned_embeds.safetensors

### Visualizing CA maps

| Name        | Notes           |
| ------------- |-------------|
|`CA_Map_visualization_resolution`: list[str] / "all" / `None` | If set, output the CA maps per frame. Can be any combination of "low", "middle" and "high", or just "all" to output them all. Recommended to perform visualization on checkpoints without `q_downsample_ratio > 1` only. Not supporting `multi_ca==True` case yet. Works for simple CA case too. Default is `None`.|
|`temporal_CA_Map_visualization_resolution`: list[str] / "all" / `None` | Same as above, but works for visualizing temporal CA layers. Default is `None`.|

## Section `noise_scheduler_kwargs`
The noise schedule setting. In default it follows SD's DDPM schedule.

## Training & inference under inversion setting (experimental)

Originally, the DM input is Concat($z, z^2_t ... z^F_t$) with $z = VAE(x_0)$ being the latent code of the reference image $x_0$. This section is concerned with a setting where, $x_0$ is encoded in the inversion results $z_T^0$, s.t. the DM input becomes a function of $z_T^0$ instead of having $z$ explicitly.

To use this feature:

### Root level

for every YAML file, put `use_DDIMInv_data: True` at root level.

### Data

First, you have to pre-compute the inversion results and store them into .pt files. One file corresponds to an image (can be a frame of a trainig video, or an image for testing).

Under `train_data`, add `DDIMInv_latent_root`, pointing to the path of pre-computed inversion latents.

### Training

Same as other losses.

For inversion setting exclusively, we have  `use_DDIMInv_first_frame_loss` to enforce first frame reconstruction. Default = `False`.

### Validation & inference

Under `validation_data` section, add `DDIMInv_latent_path`, pointing to the path of pre-computed inversion latents.
