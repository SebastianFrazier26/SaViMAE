"""Saliency-aware token masking for VideoMAE pretraining.

Salient tokens are masked at ``salient_ratio`` and background tokens at
``background_ratio``. With the defaults (0.85 / 0.95) salient regions stay
more visible than background.

VideoMAE's encoder and loss reshape ``x[~mask]`` to ``(B, -1, C)``, so every
sample in a batch must mask the same number of tokens. The generator therefore
fixes the total ``K`` per sample and splits it between the groups in proportion
to the per-group targets ``ratio * group_size``. The realised per-group ratios
equal the requested ones exactly when ``K`` equals the sum of those targets and
shift together otherwise; their ordering (salient <= background) is preserved.

Modified 2026-10-22: rewritten. The previous training path spent a fixed budget
mostly on salient tokens (the inverse of the intent), and this generator was
never used by training.
"""

from __future__ import annotations

import numpy as np


def fit_saliency_to_grid(
    saliency: np.ndarray, grid: tuple[int, int, int]
) -> np.ndarray:
    """Return saliency as a binary int8 array of exactly ``grid`` = (T, H_p, W_p).

    Time is cropped, or padded by repeating the last slice. Space must already
    match: a mismatch means the saliency was computed at a different input or
    patch size, and resizing would silently misalign tokens.
    """
    t, h, w = grid
    if saliency.ndim != 3:
        raise ValueError(
            f"saliency must be 3D [T, H_p, W_p], got shape {saliency.shape}"
        )
    if saliency.shape[1:] != (h, w):
        raise ValueError(
            f"saliency spatial grid {saliency.shape[1:]} != token grid {(h, w)}; "
            "recompute saliency patches with the training input/patch size"
        )
    if saliency.shape[0] == 0:
        raise ValueError("saliency has zero time slices")
    if saliency.shape[0] < t:
        pad = np.repeat(saliency[-1:], t - saliency.shape[0], axis=0)
        saliency = np.concatenate([saliency, pad], axis=0)
    return (saliency[:t] >= 0.5).astype(np.int8)


def allocate_masked_counts(
    n_salient: int,
    n_background: int,
    salient_ratio: float,
    background_ratio: float,
    total_masked: int,
) -> tuple[int, int]:
    """Split ``total_masked`` into (salient, background) token counts.

    Proportional to ``ratio * group_size``, then clamped to group sizes with any
    overflow moved to the other group. An empty group gets zero and the other
    group takes the whole budget.
    """
    n = n_salient + n_background
    if not 0 <= total_masked <= n:
        raise ValueError(f"total_masked={total_masked} outside [0, {n}]")
    want_s = salient_ratio * n_salient
    want_b = background_ratio * n_background
    if want_s + want_b == 0:
        share = n_salient / n if n else 0.0
    else:
        share = want_s / (want_s + want_b)
    m_s = int(round(total_masked * share))
    m_s = min(max(m_s, total_masked - n_background), min(n_salient, total_masked))
    return m_s, total_masked - m_s


class SaliencyMaskingGenerator:
    """Draws a flat ``(T * H_p * W_p,)`` int mask (1 = masked) for one clip.

    Args:
        input_size: token grid ``(T, H_p, W_p)``, i.e. ``args.window_size``.
        salient_ratio: target fraction of salient tokens that are masked.
        background_ratio: target fraction of non-salient tokens that are masked.
        saliency_patches: ``[T_s, H_p, W_p]`` saliency, binarised at 0.5.
        total_mask_ratio: overall fraction of tokens masked (fixes ``K``).
            Defaults to ``background_ratio`` (the larger ratio), so the number
            of visible tokens matches the earlier 0.95 budget.
    """

    def __init__(
        self,
        input_size: tuple[int, int, int],
        salient_ratio: float,
        background_ratio: float,
        saliency_patches: np.ndarray,
        total_mask_ratio: float | None = None,
    ) -> None:
        for name, r in (
            ("salient_ratio", salient_ratio),
            ("background_ratio", background_ratio),
        ):
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {r}")
        if total_mask_ratio is None:
            total_mask_ratio = background_ratio
        if not 0.0 <= total_mask_ratio <= 1.0:
            raise ValueError(
                f"total_mask_ratio must be in [0, 1], got {total_mask_ratio}"
            )

        self.grid = tuple(int(x) for x in input_size)
        self.salient_ratio = float(salient_ratio)
        self.background_ratio = float(background_ratio)
        self.total_patches = int(np.prod(self.grid))
        # At least one masked and one visible token, or the loss/encoder are empty.
        k = int(round(total_mask_ratio * self.total_patches))
        self.total_masked = max(1, min(self.total_patches - 1, k))

        flat = fit_saliency_to_grid(saliency_patches, self.grid).reshape(-1)
        self.salient_indices = np.flatnonzero(flat == 1)
        self.background_indices = np.flatnonzero(flat == 0)
        self.n_salient_masked, self.n_background_masked = allocate_masked_counts(
            len(self.salient_indices),
            len(self.background_indices),
            self.salient_ratio,
            self.background_ratio,
            self.total_masked,
        )

    def __repr__(self) -> str:
        return (
            f"SaliencyMaskingGenerator(grid={self.grid}, salient_ratio={self.salient_ratio}, "
            f"background_ratio={self.background_ratio}, total_masked={self.total_masked})"
        )

    def __call__(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        mask = np.zeros(self.total_patches, dtype=np.int64)
        for indices, count in (
            (self.salient_indices, self.n_salient_masked),
            (self.background_indices, self.n_background_masked),
        ):
            if count:
                mask[rng.choice(indices, size=count, replace=False)] = 1
        return mask
