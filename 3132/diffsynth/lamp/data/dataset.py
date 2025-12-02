import decord
decord.bridge.set_bridge('torch')
import os
import cv2
import numpy as np

from torch.utils.data import Dataset
from einops import rearrange
import random
import torchvision.transforms.v2 as transforms
import torch
import json

from typing import Iterable

class LAMPDataset(Dataset):
    def __init__(
            self,
            video_root: str,
            mask_root: str = None,
            prompt: str = None,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            aug: str = "flip",
            num_iterations_per_data: int = 1,
    ):
        self.video_root = video_root
        self.mask_root = mask_root
        self.video_path = []
        self.mask_path = []
        self.prompt = []
        # for video dataset with multiple prompts instead of one single prompt
        if os.path.isfile(video_root) and video_root.endswith('.json'):
            with open(video_root, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.video_path.append(item['vid'])
                    self.prompt.append(item['cap'])
        elif os.path.isdir(video_root):
            if prompt is None:
                print("Error: Prompt is needed!")
                exit()
            for video_name in os.listdir(video_root):
                if not video_name.endswith('.mp4') and not video_name.endswith('.avi'): continue
                if mask_root is not None:
                    if os.path.exists(os.path.join(self.mask_root, video_name[:-4])):
                        self.mask_path.append(os.path.join(self.mask_root, video_name[:-4]))
                        self.video_path.append(os.path.join(self.video_root, video_name))
                        self.prompt.append(prompt)
                else:
                    self.video_path.append(os.path.join(self.video_root, video_name))
                    self.prompt.append(prompt)
                # if video_root.endswith('/'):
                #     prompt = video_root.split('/')[-2].replace('_', ' ')
                # else:
                #     prompt = video_root.split('/')[-1].replace('_', ' ')
                self.prompt.append(prompt)
        else:
            if prompt is None:
                print("Error: Prompt is needed!")
                exit()
            self.video_path.append(video_root)
            # self.prompt.append(video_root.split('/')[-1].replace('_', ' ').replace('.mp4', ''))
            self.prompt.append(prompt)

        # self.prompt_ids = []

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        # Augmentation
        T = []
        if 'flip' in aug:
            T.append(transforms.RandomHorizontalFlip())
        if 'crop' in aug:
            T.append(transforms.RandomResizedCrop((height, width), (0.75, 1), ratio=(1.0, 1.0)))
        if 'color' in aug:
            T.append(transforms.ColorJitter(brightness=0.5, saturation=0.2, hue=0.05, contrast=0.2))
        T.append(transforms.ToTensor())
        self.transforms = transforms.Compose(T)

        self.num_iterations_per_data = num_iterations_per_data
        if num_iterations_per_data > 1:
            n = num_iterations_per_data
            self.video_path = [v for v in self.video_path for _ in range(n)]
            if mask_root is not None:
                self.mask_path = [m for m in self.mask_path for _ in range(n)]
            self.prompt = [p for p in self.prompt for _ in range(n)]
            self.shuffle()

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, index):
        vr = decord.VideoReader(self.video_path[index], width=self.width, height=self.height)

        if self.mask_root is not None: 
            frame_path = []
            frame_idx = []
            root = self.mask_path[index]
            for frame_name in os.listdir(root):
                if frame_name.endswith(".jpg"):
                    path = os.path.join(root, frame_name)
                    frame_path.append(path)
                    frame_idx.append(int(frame_name.split('_')[0]))

            frame_path.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))
            frame_idx.sort()

            copied_frame_path = []
            j = 0
            lst_idx = 0

            for i in range(frame_idx[-1]):
                if frame_idx[j] > i:
                    copied_frame_path.append(frame_path[lst_idx])
                elif frame_idx[j] == i:
                    j += 1
                    lst_idx = j
                    copied_frame_path.append(frame_path[lst_idx])

        n_frames = len(vr)
        if self.mask_root is not None:
            n_frames = len(copied_frame_path)
        max_frame_rate = (n_frames-1) // (self.n_sample_frames-1)
        
        if isinstance(self.sample_frame_rate, Iterable):
            # log (241227): in this case, sample_frame_rate = [fps_min, fps_max]
            fps_min, fps_max = self.sample_frame_rate

            if max_frame_rate < fps_min:
                print(f"[Dataset] WARNING: Current video too short ({n_frames} frames). Fps set to {max_frame_rate}.")
                sample_frame_rate = max_frame_rate
            elif max_frame_rate < fps_max:
                print(f"[Dataset] WARNING: Current video too short ({n_frames} frames). Fps_max set to {max_frame_rate}.")
                sample_frame_rate = random.randint(fps_min, max_frame_rate)
            else:
                sample_frame_rate = random.randint(fps_min, fps_max)
        else:
            sample_frame_rate = min(self.sample_frame_rate, max_frame_rate)

        start_idx = random.randint(0, n_frames- (self.n_sample_frames-1) * sample_frame_rate-1)
        sample_index = list(range(start_idx, n_frames, sample_frame_rate))[:self.n_sample_frames]

        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> c f h w") # log 250328: Wan expects cfhw instead of fchw
        # augmentation
        if self.mask_root is not None:
            masks_bin = load_masks([copied_frame_path[i] for i in sample_index], self.width, self.height)
            masks_rgb = load_masks_rgb([copied_frame_path[i] for i in sample_index], self.width, self.height)
            masks_rgb = rearrange(masks_rgb, "f h w c -> c f h w")
            # augmentation

            masks_rgb = torch.tensor(masks_rgb)

            out = self.transforms({'video': video, 'masks_bin': masks_bin, "masks_rgb": masks_rgb})
            video = out["video"]
            masks_bin = out["masks_bin"]
            masks_rgb = out["masks_rgb"]
        else:
            video = self.transforms(video)

        example = {
            "text": self.prompt[index],
            "video": (video / 127.5 - 1.0), 
            "path": self.video_path[index]
        }
        if self.mask_root is not None:
            example["masks_bin"] = masks_bin
            example["masks_rgb"] = masks_rgb

        return example
    
    def shuffle(self):
        n = self.num_iterations_per_data
        self.video_path = self.video_path[::n]
        self.prompt = self.prompt[::n]
        if self.mask_root is not None:
            self.mask_path = self.mask_path[::n]
            combined = list(zip(self.video_path, self.mask_path, self.prompt))
            random.shuffle(combined)
            self.video_path, self.mask_path, self.prompt = zip(*combined)
        else:
            combined = list(zip(self.video_path, self.prompt))
            random.shuffle(combined)
            self.video_path, self.prompt = zip(*combined)

        self.video_path = [v for v in self.video_path for _ in range(n)]
        self.prompt = [p for p in self.prompt for _ in range(n)]
        if self.mask_root is not None:
            self.mask_path = [m for m in self.mask_path for _ in range(n)]

def load_masks(frame_paths, width, height):
    return torch.stack([
                torch.tensor(
                    cv2.resize(cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE), (width // 8, height // 8), interpolation=cv2.INTER_AREA) / 255.0,
                    dtype=torch.float16,
                    device="cuda"
                )
                for frame_path in frame_paths
            ])

def load_masks_rgb(frame_paths, width, height):
    frames = [
        cv2.cvtColor(cv2.resize(cv2.imread(frame_path), (width, height)), cv2.COLOR_BGR2RGB)
        for frame_path in frame_paths
    ]

    new = np.stack(frames)
    return new

def load_frames(frame_paths, width, height):
    return torch.stack([
                torch.tensor(
                    cv2.cvtColor(cv2.resize(cv2.imread(frame_path), (width, height)), cv2.COLOR_BGR2RGB)
                )
                for frame_path in frame_paths
            ])