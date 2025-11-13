# VideoMAE/datasets.py

import os
import numpy as np
from torchvision import transforms

from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoMAE

# NEW: import our saliency-aware generator
from SaVi.saliency_masking_generator import SaliencyAwareTubeMaskingGenerator


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


class DataAugmentationForSaViMAE(object):
    """
    Saliency-aware augmentation.

    Assumes:
    - we know the video_id so we can load `saliency_root/{video_id}.npy`
    - saliency_patches.npy has shape [T, H_p, W_p]

    __call__ now expects: (images, video_id)
    """

    def __init__(self, args):
        self.args = args
        self.saliency_root = args.saliency_root
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

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

    def _load_saliency_patches(self, video_id: str):
        """
        video_id: string key derived from dataset; must match .npy filename.
        Returns saliency_patches [T, H_p, W_p]
        """
        fname = os.path.join(self.saliency_root, f"{video_id}.npy")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Saliency file not found: {fname}")
        saliency = np.load(fname)  # [T, H_p, W_p]
        return saliency

    def __call__(self, data):
        """
        data: tuple (images, video_id)
        images: list of PIL images or frames
        video_id: string identifier
        """
        images, video_id = data
        process_data, _ = self.transform(images)

        # Load patch-level saliency for this video
        saliency_patches = self._load_saliency_patches(video_id)

        # Build saliency-aware generator
        gen = SaliencyAwareTubeMaskingGenerator(
            input_size=self.args.window_size,
            salient_ratio=self.args.salient_mask_ratio,
            nonsalient_ratio=self.args.nonsalient_mask_ratio,
            saliency_patches=saliency_patches,
        )

        bool_mask = gen()  # np.array [T*H_p*W_p]
        return process_data, bool_mask

    def __repr__(self):
        repr_str = "(DataAugmentationForSaViMAE,\n"
        repr_str += " transform = %s,\n" % str(self.transform)
        repr_str += " saliency_root = %s,\n" % str(self.saliency_root)
        repr_str += ")"
        return repr_str


def build_pretraining_dataset(args):
    """
    Choose between baseline VideoMAE augmentation and SaVi augmentation.
    """

    if args.mask_type == 'savi':
        transform = DataAugmentationForSaViMAE(args)
    else:
        transform = DataAugmentationForVideoMAE(args)

    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
    )

    print("Data Aug = %s" % str(transform))
    return dataset
