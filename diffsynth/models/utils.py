import torch, os
from safetensors import safe_open
from contextlib import contextmanager
import hashlib

def estimate_x0(xt, t_id, scheduler, epsilon_pred):
    """
    Estimate x0 using the VP formulation
    """
    # alpha_prod_t = scheduler.alphas_cumprod[t].to(xt.device)
    # alpha_t = torch.sqrt(alpha_prod_t).view(-1, 1, 1, 1)
    # sigma_t = torch.sqrt(1 - alpha_prod_t).view(-1, 1, 1, 1)
    # x0_pred = (xt - sigma_t * epsilon_pred) / alpha_t

    sigma_t = scheduler.sigmas[t_id].view(-1, 1, 1, 1).to(xt.device)
    x0_pred = (xt - sigma_t * epsilon_pred) / (1 - sigma_t)
    return x0_pred

def CalculateDiceLoss(input, target):
    eps = 1e-6

    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                            .format(input.shape))
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} {}"
                            .format(input.shape, target.shape))
    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} {}" .format(
                input.device, target.device))
    
    # compute softmax over the classes axis

    # log (23-05-24): discard first(background) channel after softmax, then calculate IoU with the rest, 
    # fix blank seg result. (same at line 73)
    # input_soft = F.softmax(input, dim=1)
    # print(f"{torch.unique(input_soft)} {torch.unique(target)}")
    # input_soft = input_soft[:, 1:, :, :]

    # # create the labels one hot tensor
    # # target_one_hot = one_hot(target, num_classes=input.shape[1],
    # #                          device=input.device, dtype=input.dtype)
    
    # target_one_hot = torch.eye(input.shape[1], device=target.device)[target.squeeze(1).long()]
    # target_one_hot = target_one_hot.permute(0, 3, 1, 2).cuda().float()
    # target_one_hot = target_one_hot[:, 1:, :, :]

    # compute the actual dice score
    dims = (1, 2, 3)
    target = target.float()
    input = input.float()
    intersection = torch.sum(input * target, dims)
    cardinality = torch.sum(input + target, dims)

    dice_score = 2. * intersection / (cardinality + eps)
    # print("Loss: ", (1. - dice_score).shape)

    return (1. - dice_score)

def gaussian_kernel(size, sigma, device):
    """Generate a 2D Gaussian kernel."""
    coords = torch.arange(size, device=device) - size // 2
    grid = coords[:, None]**2 + coords[None, :]**2
    kernel = torch.exp(-grid / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize
    return kernel

def gaussian_blur(x, kernel_size, sigma):
    """Apply Gaussian blur to a 3D tensor (B, H, W) on GPU."""
    # Generate Gaussian kernel
    device = x.device
    kernel = gaussian_kernel(kernel_size, sigma, device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)  # Shape for convolution
    
    # Add channel dimension and apply convolution
    x = x.unsqueeze(1)  # Shape (B, 1, H, W)
    x_blurred = F.conv2d(x, kernel, padding=kernel_size // 2)
    return x_blurred.squeeze(1)  # Remove channel dimension

@contextmanager
def init_weights_on_device(device = torch.device("meta"), include_buffers :bool = False):
    
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
            
    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper
    
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}
    
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in [
            "safetensors", "bin", "ckpt", "pth", "pt"
        ]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-3:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-3:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape)>=1 and source_state_dict[name].shape[0]%3==0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(search_parameter(source_state_dict[name][i*length: i*length+length], target_state_dict))
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()