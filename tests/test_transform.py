import logging

import numpy as np
import pytest
import torch

from SaVi.saliency_io import (
    FallbackCounter,
    SaliencyStore,
    SaliencyUnavailableError,
    saliency_path,
    uniform_frame_indices,
)
from SaVi.transform import DataAugmentationForSaViMAE, tubelet_saliency

PATCH = 4  # tiny patches keep tensors small; grid logic is size-independent


class FakeBase:
    """Stands in for SimpleVideoMAETransform: list of frames -> (C, T, H, W)."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, sample):
        images, label = sample
        return torch.zeros(3, len(images), self.size, self.size), label


def write_saliency(root, video_id, arr):
    path = saliency_path(root, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))
    return path


def make_transform(root, fallback="error", seed=0, total=None, tubelet=2, size=16):
    return DataAugmentationForSaViMAE(
        FakeBase(size),
        SaliencyStore(root, fallback=fallback, counter=FallbackCounter()),
        salient_ratio=0.5,
        background_ratio=0.9,
        total_mask_ratio=total,
        tubelet_size=tubelet,
        patch_size=PATCH,
        seed=seed,
    )


def frames(n=8):
    return [object()] * n


def left_half_saliency(t_source=20, hw=4):
    sal = np.zeros((t_source, hw, hw), dtype=np.float32)
    sal[:, :, : hw // 2] = 1
    return sal


def test_saliency_path_uses_last_two_components_for_any_prefix(tmp_path):
    expected = tmp_path / "Cls" / "v_1.npy"
    assert saliency_path(tmp_path, "datasets/UCF-101-frames/Cls/v_1") == expected
    assert saliency_path(tmp_path, "/abs/hpc/home/x/Cls/v_1") == expected


def test_saliency_root_is_honoured(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_saliency(a, "x/Cls/v1", np.ones((8, 4, 4)))  # all salient
    write_saliency(b, "x/Cls/v1", np.zeros((8, 4, 4)))  # none salient
    tf_a, tf_b = make_transform(a), make_transform(b)
    # Only root "a" has salient tokens, so the group split must differ.
    _, mask_a = tf_a((frames(), "x/Cls/v1"))
    _, mask_b = tf_b((frames(), "x/Cls/v1"))
    assert mask_a.sum() == mask_b.sum()  # same fixed K
    store_a = tf_a.store.load("x/Cls/v1")
    store_b = tf_b.store.load("x/Cls/v1")
    assert store_a.all() and not store_b.any()


def test_mask_shape_matches_token_grid_for_tubelet_sizes(tmp_path):
    for tubelet in (1, 2, 4):
        write_saliency(tmp_path, "x/Cls/v1", np.zeros((8, 4, 4)))
        tf = make_transform(tmp_path, tubelet=tubelet)
        video, mask = tf((frames(8), "x/Cls/v1"))
        assert video.shape == (3, 8, 16, 16)
        assert mask.shape == ((8 // tubelet) * 4 * 4,)
        assert mask.dtype == torch.int64


def test_indivisible_video_shape_is_rejected(tmp_path):
    write_saliency(tmp_path, "x/Cls/v1", np.zeros((8, 4, 4)))
    with pytest.raises(ValueError, match="not divisible"):
        make_transform(tmp_path, tubelet=3)((frames(8), "x/Cls/v1"))


def test_saliency_grid_mismatch_is_an_error_even_with_fallback(tmp_path):
    write_saliency(tmp_path, "x/Cls/v1", np.zeros((8, 7, 7)))
    with pytest.raises(ValueError, match="spatial grid"):
        make_transform(tmp_path, fallback="random")((frames(8), "x/Cls/v1"))


def test_salient_group_masked_less_through_full_transform(tmp_path):
    write_saliency(tmp_path, "x/Cls/v1", left_half_saliency())
    tf = make_transform(tmp_path, total=0.7)
    _, mask = tf((frames(8), "x/Cls/v1"))
    grid = mask.numpy().reshape(4, 4, 4)
    assert grid[:, :, :2].mean() < grid[:, :, 2:].mean()
    assert int(mask.sum()) == round(0.7 * 64)


def test_tubelet_saliency_samples_clip_frames_and_unions_pairs():
    sal = np.zeros((10, 2, 2), dtype=np.float32)
    idx = uniform_frame_indices(10, 4)  # [0, 3, 6, 9]
    sal[idx[1], 0, 0] = 1  # second frame of the first tubelet
    sal[2, 1, 1] = 1  # a frame the clip never samples
    out = tubelet_saliency(sal, num_frames=4, tubelet_size=2)
    assert out.shape == (2, 2, 2)
    assert out[0, 0, 0] and not out[:, 1, 1].any() and not out[1].any()


def test_mask_deterministic_per_seed_epoch_and_video(tmp_path):
    write_saliency(tmp_path, "x/Cls/v1", left_half_saliency())
    write_saliency(tmp_path, "x/Cls/v2", left_half_saliency())

    def draw(tf, vid):
        return tf((frames(8), vid))[1].numpy()

    tf1, tf2 = make_transform(tmp_path, seed=3), make_transform(tmp_path, seed=3)
    assert np.array_equal(draw(tf1, "x/Cls/v1"), draw(tf2, "x/Cls/v1"))
    assert not np.array_equal(draw(tf1, "x/Cls/v1"), draw(tf1, "x/Cls/v2"))
    before = draw(tf1, "x/Cls/v1")
    tf1.set_epoch(1)
    assert not np.array_equal(before, draw(tf1, "x/Cls/v1"))
    assert not np.array_equal(
        before, draw(make_transform(tmp_path, seed=4), "x/Cls/v1")
    )


def test_missing_file_errors_by_default(tmp_path):
    with pytest.raises(SaliencyUnavailableError, match="saliency_fallback"):
        make_transform(tmp_path)((frames(8), "x/Cls/nope"))


def test_corrupt_file_errors_by_default(tmp_path):
    path = saliency_path(tmp_path, "x/Cls/bad")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not an npy file")
    with pytest.raises(SaliencyUnavailableError):
        make_transform(tmp_path)((frames(8), "x/Cls/bad"))


def test_wrong_ndim_file_is_treated_as_corrupt(tmp_path):
    write_saliency(tmp_path, "x/Cls/flat", np.zeros(64))
    with pytest.raises(SaliencyUnavailableError):
        make_transform(tmp_path)((frames(8), "x/Cls/flat"))


def test_random_fallback_warns_counts_and_still_masks(tmp_path, caplog):
    tf = make_transform(tmp_path, fallback="random", total=0.75)
    with caplog.at_level(logging.WARNING, logger="SaVi.saliency_io"):
        _, mask = tf((frames(8), "x/Cls/nope"))
    assert int(mask.sum()) == round(0.75 * 64)
    assert tf.store.counter.value == 1
    assert "using random masking" in caplog.text
    assert tf.store.counter.reset() == 1 and tf.store.counter.value == 0


def test_store_rejects_unknown_fallback(tmp_path):
    with pytest.raises(ValueError):
        SaliencyStore(tmp_path, fallback="ignore")
