# SaVi/saliency_masking_generator.py

import numpy as np

class SaliencyMaskingGenerator:
    """
    Saliency-aware version of TubeMaskingGenerator.

    It uses *different* mask ratios for salient vs non-salient patches.
    - salient_ratio: fraction of salient patches to mask
    - nonsalient_ratio: fraction of non-salient patches to mask

    input_size: (T, H_p, W_p) = args.window_size from run_mae_pretraining.
    """

    def __init__(
        self,
        input_size,
        salient_ratio: float,
        nonsalient_ratio: float,
        saliency_patches: np.ndarray,
    ):
        self.frames, self.height, self.width = input_size
        self.salient_ratio = float(salient_ratio)
        self.nonsalient_ratio = float(nonsalient_ratio)

        # saliency_patches shape: [T_total, H_p, W_p]
        assert saliency_patches.ndim == 3
        # Use first self.frames frames; if fewer, pad last frame
        if saliency_patches.shape[0] < self.frames:
            pad_n = self.frames - saliency_patches.shape[0]
            pad_frames = np.repeat(saliency_patches[-1:], pad_n, axis=0)
            saliency_patches = np.concatenate([saliency_patches, pad_frames], axis=0)
        saliency_patches = saliency_patches[: self.frames]

        # binarize in case there are floats
        self.saliency = (saliency_patches >= 0.5).astype(np.int32)  # [T, H_p, W_p]

        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame

        # Flatten indices
        # index scheme: t * (H_p*W_p) + i * W_p + j
        flat_saliency = self.saliency.reshape(-1)  # [T*H_p*W_p]
        self.salient_indices = np.where(flat_saliency == 1)[0]
        self.nonsalient_indices = np.where(flat_saliency == 0)[0]

    def __repr__(self):
        return (
            f"SaliencyMaskingGenerator("
            f"total_patches={self.total_patches}, "
            f"salient_ratio={self.salient_ratio}, "
            f"nonsalient_ratio={self.nonsalient_ratio})"
        )

    def __call__(self):
        mask = np.zeros(self.total_patches, dtype=np.int32)

        # How many from each group do we mask?
        n_salient = len(self.salient_indices)
        n_nonsalient = len(self.nonsalient_indices)

        num_salient_to_mask = int(self.salient_ratio * n_salient)
        num_nonsalient_to_mask = int(self.nonsalient_ratio * n_nonsalient)

        # Sample indices
        if n_salient > 0 and num_salient_to_mask > 0:
            chosen_sal = np.random.choice(
                self.salient_indices, size=num_salient_to_mask, replace=False
            )
            mask[chosen_sal] = 1

        if n_nonsalient > 0 and num_nonsalient_to_mask > 0:
            chosen_nonsal = np.random.choice(
                self.nonsalient_indices, size=num_nonsalient_to_mask, replace=False
            )
            mask[chosen_nonsal] = 1

        # Return as (T*H_p*W_p,) for consistency with original generator
        return mask