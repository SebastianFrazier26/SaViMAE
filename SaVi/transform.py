"""Dataset transform: base VideoMAE transform + saliency-aware token mask."""

from __future__ import annotations

import zlib

import numpy as np
import torch

from SaVi.saliency_io import SaliencyStore, uniform_frame_indices
from SaVi.saliency_masking_generator import SaliencyMaskingGenerator


def tubelet_saliency(
    saliency: np.ndarray, num_frames: int, tubelet_size: int
) -> np.ndarray:
    """Reduce per-source-frame saliency ``[T_s, H_p, W_p]`` to tubelet tokens.

    Selects the saliency slice of each frame the clip samples, then a tubelet
    is salient if any of its frames is. Returns ``[num_frames // tubelet, H_p, W_p]``.
    """
    idx = uniform_frame_indices(saliency.shape[0], num_frames)
    per_frame = saliency[idx] >= 0.5
    t_tok = num_frames // tubelet_size
    grouped = per_frame.reshape(t_tok, tubelet_size, *per_frame.shape[1:])
    return grouped.any(axis=1)


class DataAugmentationForSaViMAE:
    """Maps ``(images, video_id)`` to ``(video (C,T,H,W), bool_masked_pos (N,))``.

    The mask is a flat token mask over the grid
    ``(T // tubelet_size, H // patch_size, W // patch_size)`` with a fixed
    number of masked tokens (see saliency_masking_generator).

    With ``seed`` set, the mask for a video is a pure function of
    (seed, epoch, video_id), so it is reproducible across worker counts yet
    still changes every epoch (call ``set_epoch``). With ``seed=None`` it is
    freshly random.
    """

    def __init__(
        self,
        transform,
        store: SaliencyStore,
        salient_ratio: float,
        background_ratio: float,
        total_mask_ratio: float | None = None,
        tubelet_size: int = 2,
        patch_size: int = 16,
        seed: int | None = None,
    ) -> None:
        self.transform = transform
        self.store = store
        self.salient_ratio = float(salient_ratio)
        self.background_ratio = float(background_ratio)
        self.total_mask_ratio = total_mask_ratio
        self.tubelet_size = int(tubelet_size)
        self.patch_size = int(patch_size)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, video_id: str) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(
            [self.seed, self.epoch, zlib.crc32(video_id.encode())]
        )

    def __call__(self, sample):
        images, video_id = sample
        video, _ = self.transform((images, 0))  # (C, T, H, W)
        _, t, h, w = video.shape
        if t % self.tubelet_size or h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"video shape (T={t}, H={h}, W={w}) not divisible by "
                f"tubelet_size={self.tubelet_size}, patch_size={self.patch_size}"
            )
        grid = (t // self.tubelet_size, h // self.patch_size, w // self.patch_size)

        raw = self.store.load(video_id)
        if raw is None:
            # Random fallback: an all-background volume masks uniformly at K.
            tokens = np.zeros(grid, dtype=bool)
        else:
            tokens = tubelet_saliency(raw, t, self.tubelet_size)

        generator = SaliencyMaskingGenerator(
            grid,
            self.salient_ratio,
            self.background_ratio,
            tokens,
            self.total_mask_ratio,
        )
        return video, torch.from_numpy(generator(self._rng(video_id)))
