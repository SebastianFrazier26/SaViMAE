# VideoMAE/datasets.py
# Modified for SaViMAE (2026-10-22): the saliency-aware transform moved to
# SaVi/transform.py; this module wires it in and loads UCF-101 frame folders.
# Dropped the unused `.kinetics` import (it pulled in decord and cv2 for nothing).
# Remainder of VideoMAE/ is upstream MCG-NJU/VideoMAE, CC BY-NC 4.0 (see LICENSE).
import glob
import os

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset
from .transforms import *
from .masking_generator import TubeMaskingGenerator

from SaVi.saliency_io import SaliencyStore, FallbackCounter, uniform_frame_indices
from SaVi.transform import DataAugmentationForSaViMAE

class DataAugmentationForVideoMAE(object):
    """
    Original VideoMAE augmentation + tube masking.
    Kept for baseline runs (mask_type='tube').
    """

    def __init__(self, args):
        self.args = args
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]   # IMAGENET_DEFAULT_STD

        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(
            args.input_size, [1, .875, .75, .66]
        )

        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])

        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        else:
            self.masked_position_generator = None

    def __call__(self, images):
        process_data, _ = self.transform(images)
        if self.masked_position_generator is None:
            raise RuntimeError("masked_position_generator is None for mask_type != 'tube'")
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr_str = "(DataAugmentationForVideoMAE,\n"
        repr_str += " transform = %s,\n" % str(self.transform)
        repr_str += " Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr_str += ")"
        return repr_str


class UCFPretrainDataset(Dataset):
    """
    Minimal dataset for SaViMAE pretraining on the UCF-101 frames subset.

    Expects a list file with lines:
        <video_frames_dir> <label>

    Example line:
        datasets/UCF-101-frames-subset/ApplyLipstick/v_ApplyLipstick_g14_c01 0
    """

    def __init__(self, list_file, num_frames, sampling_rate, transform=None):
        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.split()
                self.samples.append((path, int(label)))

        self.num_frames = num_frames
        self.sampling_rate = sampling_rate  # unused: frames are sampled uniformly
        self.transform = transform

    def set_epoch(self, epoch):
        # Per-epoch masks; workers are re-created each epoch, so they see this.
        if hasattr(self.transform, "set_epoch"):
            self.transform.set_epoch(epoch)

    def pop_fallback_count(self):
        """Videos that used random masking since the last call (0 if no store)."""
        store = getattr(self.transform, "store", None)
        return store.counter.reset() if store is not None else 0

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_dir):
        """
        Load frames from img_XXXXX.jpg in video_dir and sample num_frames of them.
        """
        frame_files = sorted(glob.glob(os.path.join(video_dir, "img_*.jpg")))
        if len(frame_files) == 0:
            raise RuntimeError(f"No frames found in {video_dir}")

        # Same indices SaVi.transform uses to pick each tubelet's saliency.
        indices = uniform_frame_indices(len(frame_files), self.num_frames)
        frames = [Image.open(frame_files[i]).convert("RGB") for i in indices]
        return frames

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        images = self._load_frames(video_dir)
        video_id = video_dir  # use full path as ID; used to locate saliency

        if self.transform is not None:
            videos, bool_masked_pos = self.transform((images, video_id))
        else:
            # Should not really happen in pretraining, but keep a fallback
            raise RuntimeError("UCFPretrainDataset requires a transform returning (videos, bool_masked_pos)")

        # We ignore label for MAE pretraining (it’s self-supervised)
        return videos, bool_masked_pos


class SimpleVideoMAETransform(object):
    """
    Very simple spatial transform for VideoMAE-style inputs.

    Input: (images, label)
        images: list of PIL images (len T)
        label : int (we just forward it)

    Output: (video_tensor, label)
        video_tensor: torch.Tensor of shape (C, T, H, W)
    """

    def __init__(self, input_size):
        self.resize = T.Resize((input_size, input_size))
        self.to_tensor = T.ToTensor()
        # Standard ImageNet normalization
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __call__(self, sample):
        images, label = sample  # images: list of PIL.Image

        processed = []
        for img in images:
            img = self.resize(img)
            img = self.to_tensor(img)
            img = self.normalize(img)
            processed.append(img)  # (C, H, W)

        # Stack along time dimension -> (T, C, H, W)
        video = torch.stack(processed, dim=0)
        # Permute to (C, T, H, W) which VideoMAE expects
        video = video.permute(1, 0, 2, 3)

        return video, label

def build_pretraining_dataset(args):
    """
    Build the pretraining dataset for SaViMAE on UCF-101 frame folders.

    Expected args: data_path, input_size, num_frames, sampling_rate, mask_type,
    and for mask_type 'savi': saliency_root, salient_mask_ratio,
    background_mask_ratio, and optionally savi_total_mask_ratio,
    saliency_fallback, seed, tubelet_size, patch_size.
    """
    if getattr(args, "mask_type", None) != "savi":
        # The plain-transform path returned (video, label) where the engine
        # expects (video, bool_masked_pos), so it could never train.
        raise ValueError(
            f"build_pretraining_dataset only supports --mask_type savi, got {args.mask_type!r}"
        )
    if not getattr(args, "saliency_root", None):
        raise ValueError("--saliency_root is required with --mask_type savi")

    patch_size = getattr(args, "patch_size", (16, 16))
    transform = DataAugmentationForSaViMAE(
        transform=SimpleVideoMAETransform(input_size=args.input_size),
        store=SaliencyStore(
            args.saliency_root,
            fallback=getattr(args, "saliency_fallback", "error"),
            counter=FallbackCounter(),
        ),
        salient_ratio=args.salient_mask_ratio,
        background_ratio=args.background_mask_ratio,
        total_mask_ratio=getattr(args, "savi_total_mask_ratio", None),
        tubelet_size=getattr(args, "tubelet_size", 2),
        patch_size=patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size,
        seed=getattr(args, "seed", None),
    )

    dataset = UCFPretrainDataset(
        list_file=args.data_path,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        transform=transform,
    )

    print(f"[build_pretraining_dataset] Loaded {len(dataset)} videos from {args.data_path}")
    return dataset
