import numpy as np
import pytest

from SaVi.saliency_masking_generator import (
    SaliencyMaskingGenerator,
    allocate_masked_counts,
    fit_saliency_to_grid,
)

GRID = (8, 14, 14)
N = 8 * 14 * 14


def block_saliency(fraction: float) -> np.ndarray:
    """Salient block in the top rows of every slice, ~fraction of tokens."""
    sal = np.zeros(GRID, dtype=np.float32)
    sal[:, : round(14 * fraction), :] = 1.0
    return sal


def realised(mask: np.ndarray, sal: np.ndarray) -> tuple[float, float]:
    flat = sal.reshape(-1) >= 0.5
    return mask[flat].mean(), mask[~flat].mean()


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5])
def test_salient_masked_less_than_background(fraction):
    """The regression: salient tokens must be kept MORE visible, not less."""
    sal = block_saliency(fraction)
    gen = SaliencyMaskingGenerator(GRID, 0.85, 0.95, sal)
    sal_r, bg_r = realised(gen(np.random.default_rng(0)), sal)
    assert sal_r < bg_r
    # K defaults to 0.95*N, above the natural count, so realised ratios sit slightly
    # above the targets (about +0.05 at 50% salient) with salient < background kept.
    assert abs(sal_r - 0.85) < 0.06
    assert abs(bg_r - 0.95) < 0.06


def test_exact_group_ratios_when_total_matches_targets():
    sal = block_saliency(0.5)  # 50% salient
    gen = SaliencyMaskingGenerator(GRID, 0.8, 0.9, sal, total_mask_ratio=0.85)
    sal_r, bg_r = realised(gen(np.random.default_rng(1)), sal)
    assert sal_r == pytest.approx(0.8, abs=0.01)
    assert bg_r == pytest.approx(0.9, abs=0.01)


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_total_masked_is_fixed_regardless_of_saliency(fraction):
    gen = SaliencyMaskingGenerator(GRID, 0.85, 0.95, block_saliency(fraction))
    masks = [gen(np.random.default_rng(s)) for s in range(3)]
    assert {int(m.sum()) for m in masks} == {gen.total_masked}
    assert gen.total_masked == round(0.95 * N)
    assert masks[0].shape == (N,)


def test_deterministic_given_rng_and_varies_across_seeds():
    gen = SaliencyMaskingGenerator(GRID, 0.85, 0.95, block_saliency(0.3))
    a, b = gen(np.random.default_rng(5)), gen(np.random.default_rng(5))
    c = gen(np.random.default_rng(6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_no_salient_tokens_masks_uniformly_at_total():
    sal = np.zeros(GRID, dtype=np.float32)
    gen = SaliencyMaskingGenerator(GRID, 0.85, 0.95, sal)
    assert (gen.n_salient_masked, gen.n_background_masked) == (0, gen.total_masked)
    assert gen(np.random.default_rng(0)).sum() == gen.total_masked


def test_all_salient_tokens_uses_salient_group_only():
    sal = np.ones(GRID, dtype=np.float32)
    gen = SaliencyMaskingGenerator(GRID, 0.85, 0.95, sal)
    assert (gen.n_salient_masked, gen.n_background_masked) == (gen.total_masked, 0)
    assert gen(np.random.default_rng(0)).sum() == gen.total_masked


def test_mask_never_empties_or_fills_the_clip():
    sal = block_saliency(0.5)
    low = SaliencyMaskingGenerator(GRID, 0.0, 0.0, sal)
    high = SaliencyMaskingGenerator(GRID, 1.0, 1.0, sal)
    assert low.total_masked == 1
    assert high.total_masked == N - 1


@pytest.mark.parametrize(
    "n_s,n_b,k",
    [(0, 100, 50), (100, 0, 50), (3, 97, 95), (97, 3, 95), (50, 50, 0), (50, 50, 100)],
)
def test_allocation_respects_group_sizes_and_total(n_s, n_b, k):
    m_s, m_b = allocate_masked_counts(n_s, n_b, 0.85, 0.95, k)
    assert m_s + m_b == k
    assert 0 <= m_s <= n_s and 0 <= m_b <= n_b


def test_allocation_zero_ratios_does_not_divide_by_zero():
    assert sum(allocate_masked_counts(10, 10, 0.0, 0.0, 5)) == 5


def test_invalid_ratio_rejected():
    with pytest.raises(ValueError):
        SaliencyMaskingGenerator(GRID, 1.2, 0.95, block_saliency(0.3))


def test_fit_pads_and_crops_time_and_binarises():
    sal = np.full((3, 14, 14), 0.7, dtype=np.float32)
    out = fit_saliency_to_grid(sal, (8, 14, 14))
    assert out.shape == (8, 14, 14) and out.all()
    assert fit_saliency_to_grid(np.zeros((20, 14, 14)), (8, 14, 14)).shape == (
        8,
        14,
        14,
    )


def test_fit_rejects_spatial_mismatch():
    with pytest.raises(ValueError, match="spatial grid"):
        fit_saliency_to_grid(np.zeros((8, 7, 7)), (8, 14, 14))
