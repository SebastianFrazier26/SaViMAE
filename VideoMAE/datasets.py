# VideoMAE/datasets.py
import os
import sys
from pathlib import Path
import glob
import numpy as np
from torchvision import transforms
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset
from .transforms import *
from .masking_generator import TubeMaskingGenerator
from .kinetics import VideoMAE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../SaViMAE/VideoMAE
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                  # .../SaViMAE
_SAVI_DIR = os.path.join(_PROJECT_ROOT, "SaVi")             # adjust "SaVi" if your folder name differs

if _SAVI_DIR not in sys.path:
    sys.path.insert(0, _SAVI_DIR)
# NEW: import our saliency-aware generator
from saliency_masking_generator import SaliencyMaskingGenerator


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


# In VideoMAE/datasets.py or wherever this lives
class DataAugmentationForSaViMAE(object):
    """
    Saliency-aware wrapper around the standard VideoMAE data augmentation.

    It:
      - Applies the base VideoMAE transform to frames -> (C, T, H, W) tensor.
      - Loads per-video saliency patches from .npy.
      - Converts saliency into a token-level mask with a *fixed* number of masked
        tokens per sample (required by VideoMAE's engine_for_pretraining).
      - Returns (videos, bool_masked_pos) as expected by train_one_epoch.
    """

    def __init__(
        self,
        transform,
        saliency_root,
        salient_ratio: float,
        nonsalient_ratio: float,
        num_frames: int = 16,
        sampling_rate: int = 4,
    ):
        """
        Args:
            transform: base VideoMAE transform, expects (images, label) -> (video_tensor, label).
            saliency_root: root dir of saliency patch .npy files (used as a hint).
            salient_ratio: fraction of the *mask budget* to allocate to salient patches.
            nonsalient_ratio: overall mask ratio for the whole grid (approx; defines K).
            num_frames: number of frames fed to the model (e.g. 16).
            sampling_rate: frame sampling rate (unused here but kept for completeness).
        """
        self.transform = transform
        self.saliency_root = saliency_root
        self.salient_ratio = float(salient_ratio)
        self.nonsalient_ratio = float(nonsalient_ratio)
        self.num_frames = int(num_frames)
        self.sampling_rate = int(sampling_rate)

    def __call__(self, sample):
        images, video_id = sample

        # 1) Base VideoMAE transform: (images, label) -> (video_tensor, label)
        dummy_label = 0
        video_tensor, _ = self.transform((images, dummy_label))  # (C, T, H, W)
        _, T, H, W = video_tensor.shape

        # VideoMAE uses tubelets of temporal size 2, so number of tokens along time is T//2.
        # With patch_size=16 and input_size=224, we get H_p = W_p = 14, N_tokens = (T//2)*14*14.
        # We'll build a saliency mask of length N_tokens.
        frames_tok = T // 2

        # 2) Load saliency patches for THIS video
        saliency_path = self._get_saliency_npy_path(video_id)
        saliency_patches = self._load_saliency_weights(saliency_path)
        # saliency_patches: [T_s, H_p, W_p] (we'll adapt T_s to frames_tok)

        T_s, H_p, W_p = saliency_patches.shape

        # Ensure we have at least frames_tok frames in saliency; crop or pad as needed
        if T_s < frames_tok:
            pad_n = frames_tok - T_s
            pad_frames = np.repeat(saliency_patches[-1:], pad_n, axis=0)
            saliency_patches = np.concatenate([saliency_patches, pad_frames], axis=0)
        saliency_patches = saliency_patches[:frames_tok]  # [frames_tok, H_p, W_p]

        # 3) Convert saliency to binary and flatten to token space
        saliency_bin = (saliency_patches >= 0.5).astype(np.int32)  # 1 = salient, 0 = non-salient
        flat = saliency_bin.reshape(-1)  # length = frames_tok * H_p * W_p
        total_patches = flat.shape[0]

        salient_idx = np.where(flat == 1)[0]
        nonsalient_idx = np.where(flat == 0)[0]

        # 4) Decide how many tokens to mask: K is fixed per sample
        # Use nonsalient_ratio as the overall mask ratio (0.95 ~ heavy masking)
        K = int(self.nonsalient_ratio * total_patches)
        K = max(1, min(total_patches - 1, K))  # keep it in a sane range

        # Allocate some of K to salient tokens, rest to non-salient
        K_sal = int(self.salient_ratio * K)
        K_nonsal = K - K_sal

        rng = np.random.default_rng()

        chosen = []

        # Choose from salient set
        if len(salient_idx) > 0 and K_sal > 0:
            if len(salient_idx) < K_sal:
                K_sal = len(salient_idx)
            chosen_sal = rng.choice(salient_idx, size=K_sal, replace=False)
            chosen.append(chosen_sal)

        # Choose from nonsalient set
        if len(nonsalient_idx) > 0 and K_nonsal > 0:
            if len(nonsalient_idx) < K_nonsal:
                K_nonsal = len(nonsalient_idx)
            chosen_nonsal = rng.choice(nonsalient_idx, size=K_nonsal, replace=False)
            chosen.append(chosen_nonsal)

        if len(chosen) > 0:
            chosen = np.concatenate(chosen, axis=0)
        else:
            chosen = np.array([], dtype=np.int64)

        # If for some weird reason we didn't get enough tokens (e.g. extreme ratios),
        # fill up randomly to reach K.
        if chosen.shape[0] < K:
            remaining = np.setdiff1d(np.arange(total_patches), chosen, assume_unique=True)
            extra_need = K - chosen.shape[0]
            if remaining.shape[0] > 0:
                extra_need = min(extra_need, remaining.shape[0])
                extra = rng.choice(remaining, size=extra_need, replace=False)
                chosen = np.concatenate([chosen, extra], axis=0)

        # Final binary mask over tokens
        mask_vec = np.zeros(total_patches, dtype=np.int64)
        if chosen.shape[0] > 0:
            mask_vec[chosen] = 1  # 1 = masked

        # Convert to torch tensor (1D); DataLoader -> (B, N_tokens)
        bool_masked_pos = torch.from_numpy(mask_vec)

        # Return exactly what engine_for_pretraining expects
        return video_tensor, bool_masked_pos

    def _get_saliency_npy_path(self, video_id: str) -> str:
        """
        Map a frames video_id to the corresponding saliency .npy.

        Examples of video_id:
          'datasets/UCF-101-frames-subset/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01'
          'datasets/UCF-101-frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01'

        We want:
          'datasets/UCF101-saliency-patches-subset/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.npy'
        """
        rel = video_id

        prefixes = [
            "datasets/UCF-101-frames-subset/",
            "datasets/UCF-101-frames/",
        ]
        for prefix in prefixes:
            if rel.startswith(prefix):
                rel = rel[len(prefix):]  # e.g. 'ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01'
                break

        # You can keep using args.saliency_root if you want, but for this project
        # we know the saliency patches live here:
        saliency_root = "datasets/UCF101-saliency-patches-subset"

        return os.path.join(saliency_root, rel + ".npy")

    def _load_saliency_weights(self, saliency_path: str):
        """
        Load saliency patches and ensure they are [T, H_p, W_p].

        Also fixes any paths that still mistakenly point to UCF-101-frames*
        instead of the saliency patches directory, and falls back to a
        uniform saliency volume if the .npy truly doesn't exist.
        """
        # Fix root if it's still pointing into frames dirs
        bad_roots = [
            "datasets/UCF-101-frames-subset",
            "datasets/UCF-101-frames",
        ]
        for bad in bad_roots:
            if bad in saliency_path:
                prefix, after = saliency_path.split(bad, 1)
                saliency_root = "datasets/UCF101-saliency-patches-subset"
                saliency_path = os.path.join(prefix, saliency_root + after)
                break

        if not saliency_path.endswith(".npy"):
            saliency_path = saliency_path + ".npy"

        if not os.path.isfile(saliency_path):
            # Fallback: uniform saliency volume (all ones) for this project
            T = self.num_frames
            H_p = W_p = 14  # 224 / 16
            print(f"[WARN] Saliency file not found, using uniform saliency: {saliency_path}")
            return np.ones((T, H_p, W_p), dtype=np.float32)

        sal = np.load(saliency_path)

        # Already [T, H_p, W_p]
        if sal.ndim == 3:
            return sal

        # Flattened [T*H_p*W_p]
        if sal.ndim == 1:
            flat = sal.reshape(-1)
            total = flat.shape[0]

            if total % self.num_frames != 0:
                raise ValueError(
                    f"Cannot reshape saliency from {saliency_path}: "
                    f"total={total} not divisible by num_frames={self.num_frames}"
                )

            patches_per_frame = total // self.num_frames
            side = int(np.sqrt(patches_per_frame))
            if side * side != patches_per_frame:
                raise ValueError(
                    f"Cannot infer square patch grid for {saliency_path}: "
                    f"patches_per_frame={patches_per_frame} is not a perfect square."
                )

            sal = flat.reshape(self.num_frames, side, side)
            return sal

        raise ValueError(
            f"Unexpected saliency array shape {sal.shape} from {saliency_path}; "
            "expected 3D [T,H_p,W_p] or 1D flattened."
        )

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
        self.sampling_rate = sampling_rate
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_dir):
        """
        Load frames from img_XXXXX.jpg in video_dir and sample num_frames of them.
        """
        frame_files = sorted(glob.glob(os.path.join(video_dir, "img_*.jpg")))
        if len(frame_files) == 0:
            raise RuntimeError(f"No frames found in {video_dir}")

        # Uniformly sample num_frames indices
        # (Simple strategy; okay for our small class project run.)
        indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
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
    Build the pretraining dataset for SaViMAE on the UCF frames subset.
    """

    # Base transform: resize + to_tensor + normalize
    base_transform = SimpleVideoMAETransform(input_size=args.input_size)

    # Wrap with SaVi saliency-aware masking if requested
    if getattr(args, "mask_type", None) == "savi":
        transform = DataAugmentationForSaViMAE(
            transform=base_transform,
            saliency_root=getattr(args, "saliency_root", "datasets/UCF101-saliency-patches-subset"),
            salient_ratio=args.salient_mask_ratio,
            nonsalient_ratio=args.nonsalient_mask_ratio,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
        )
    else:
        transform = base_transform

    dataset = UCFPretrainDataset(
        list_file=args.data_path,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        transform=transform,
    )

    print(f"[build_pretraining_dataset] Loaded {len(dataset)} videos from {args.data_path}")
    return dataset