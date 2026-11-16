# Changelog

## 2026-10-22 (fix/saliency-masking)

- **Fixed saliency masking.** Training now uses `SaVi/saliency_masking_generator.py`: salient tokens masked at `--salient_mask_ratio` (0.85), background at `--background_mask_ratio` (0.95), fixed total `K` per clip (default `0.95 * N`, `--savi_total_mask_ratio` to change). The old training path masked all salient tokens when they were a minority. **Earlier results used the old behaviour and have not been rerun.**
- Honour `--saliency_root` (was hardcoded). Missing or corrupt saliency files now raise by default; `--saliency_fallback random` masks those clips randomly, logs a warning and reports a per-epoch count.
- Saliency is sampled at the frames the clip uses and combined per tubelet (was: first frames of the video). Spatial grid mismatches raise.
- Masks are seeded (`--seed`) and per-epoch; `--nonsalient_mask_ratio` kept as alias of `--background_mask_ratio`.
- Removed the unused tube/random `--mask_type` options (they could not produce a mask) and the unused `.kinetics` import in `VideoMAE/datasets.py`.
- Scripts in `scripts/` and `SaVi/compare_results.py` take paths as arguments; shared helpers in `scripts/common.py`.
- Added `pyproject.toml` + `uv.lock`, pytest suite in `tests/`, ruff, and `.github/workflows/ci.yml`.
- Site: replaced the implementation-mismatch note with a fixed-on note; regenerated `site/assets/masking.svg` (current vs previous behaviour, still illustrative).
## 2026-10-22

- Added a static project site in `site/`, deployed to GitHub Pages by `.github/workflows/pages.yml` on pushes to `main`.
- Added `site/tools/make_masking_figure.py`, which renders an illustrative masking figure on a synthetic frame from the repo's own saliency and masking code. It contains no results.
- README: added a project-site section and a licensing note (root MIT vs. `VideoMAE/` CC BY-NC 4.0). Fixed the acknowledgment that credited the VideoMAE authors as "RGBD Video SOD authors".
