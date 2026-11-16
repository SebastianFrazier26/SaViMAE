"""Locating and loading per-video patch saliency (.npy) files."""

from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path, PurePath

import numpy as np

logger = logging.getLogger(__name__)

FALLBACK_CHOICES = ("error", "random")


class SaliencyUnavailableError(RuntimeError):
    """The saliency file for a video is missing or unreadable."""


class FallbackCounter:
    """Process-shared count of videos that fell back to random masking.

    DataLoader workers hold copies of the dataset, so a plain int would never
    reach the main process. The shared Value is inherited by the workers.
    """

    def __init__(self) -> None:
        self._value = multiprocessing.Value("q", 0)

    def increment(self) -> None:
        with self._value.get_lock():
            self._value.value += 1

    @property
    def value(self) -> int:
        return int(self._value.value)

    def reset(self) -> int:
        with self._value.get_lock():
            old = int(self._value.value)
            self._value.value = 0
        return old


def uniform_frame_indices(num_source_frames: int, num_frames: int) -> np.ndarray:
    """Indices of the frames a clip is built from (uniform over the video).

    Shared by the dataset loader and the saliency lookup so a tubelet's
    saliency comes from the frames it actually contains.
    """
    return np.linspace(0, num_source_frames - 1, num_frames, dtype=int)


def saliency_path(saliency_root: str | PurePath, video_id: str) -> Path:
    """``<root>/<class>/<video>.npy`` from a frames dir ``.../<class>/<video>``.

    Uses only the last two path components, so the list file may hold relative
    or absolute (e.g. cluster) paths without any prefix rewriting.
    """
    parts = PurePath(video_id).parts
    if len(parts) < 2:
        raise ValueError(
            f"video_id {video_id!r} must look like '<...>/<class>/<video>'"
        )
    return Path(saliency_root, parts[-2], parts[-1] + ".npy")


class SaliencyStore:
    """Loads ``[T_s, H_p, W_p]`` saliency for a video from ``root``.

    On a missing or corrupt file: raises SaliencyUnavailableError when
    ``fallback == "error"``; with ``"random"`` logs a warning, bumps
    ``counter`` and returns None (the caller then masks uniformly at random).
    """

    def __init__(
        self,
        root: str | PurePath,
        fallback: str = "error",
        counter: FallbackCounter | None = None,
    ) -> None:
        if fallback not in FALLBACK_CHOICES:
            raise ValueError(
                f"fallback must be one of {FALLBACK_CHOICES}, got {fallback!r}"
            )
        self.root = Path(root)
        self.fallback = fallback
        self.counter = counter if counter is not None else FallbackCounter()

    def load(self, video_id: str) -> np.ndarray | None:
        path = saliency_path(self.root, video_id)
        try:
            # allow_pickle=False: these files are data, never code.
            arr = np.load(path, allow_pickle=False)
            if arr.ndim != 3 or arr.shape[0] == 0:
                raise ValueError(
                    f"expected non-empty [T, H_p, W_p], got shape {arr.shape}"
                )
            return arr
        except (OSError, ValueError, EOFError) as exc:
            if self.fallback == "error":
                raise SaliencyUnavailableError(
                    f"saliency unavailable for {video_id!r} at {path}: {exc}. "
                    "Use --saliency_fallback random to mask such videos at random instead."
                ) from exc
            self.counter.increment()
            logger.warning(
                "saliency unavailable at %s (%s); using random masking", path, exc
            )
            return None
