# SaViMAE — Saliency-Aware VideoMAE Pretraining

SaViMAE is a saliency-guided extension of VideoMAE designed for efficient self-supervised video representation learning.  
This repository contains:

- A simplified VideoMAE pretraining pipeline  
- Frame-based UCF101 data loading  
- Saliency map generation from raw frames  
- Patch-level saliency computation  
- A saliency-aware masking generator used by the training path (`SaVi/saliency_masking_generator.py`)  
- CPU-friendly scripts (HPC/Slurm friendly: every path is an argument, no Slurm scripts are tracked here)  

This project was built for academic experimentation and course work, emphasizing:
- clear reproducibility,
- no reliance on external RGB-D models,
- compatibility with CPU-only environments (e.g., Dartmouth Polaris),
- and a full saliency → patch → pretraining workflow.

---

## 🌐 Project site

**https://sebastianfrazier26.github.io/SaViMAE/** (pending: goes live once GitHub Pages is enabled)

The site source is in `site/` (static HTML/CSS, no build step). `.github/workflows/pages.yml` deploys it on every push to `main` and on manual dispatch. Before deploying, the workflow regenerates the illustrative masking figure from the repo code:

```bash
uv run site/tools/make_masking_figure.py   # writes site/assets/masking.svg
```

To preview locally, run `python3 -m http.server -d site 8000` and open http://localhost:8000.

**One-time setup (repo owner):** Settings → Pages → Build and deployment → Source: **GitHub Actions**.

---

## ⚖️ Licensing note

This repository has **mixed licences**, and nothing here relicenses anything:

- The root `LICENSE` (**MIT**) covers the original SaViMAE code: `SaVi/`, `scripts/` and `site/`.
- `VideoMAE/` is upstream [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) code under **CC BY-NC 4.0** (see `VideoMAE/LICENSE` and `VideoMAE/NOTICE.md`). That includes the SaViMAE modifications made inside it. Using the full training pipeline is therefore subject to its **non-commercial** terms.
- `RGBD_Video_SOD` is a git submodule that points at a separate repository with its own licence.
- UCF101 is not redistributed. Its own terms apply.

---

## 📦 Repository Structure

```
SaViMAE/
├── VideoMAE/                      # upstream VideoMAE (CC BY-NC 4.0) + SaViMAE edits
│   ├── datasets.py                # frame-folder dataset; wires in SaVi/transform.py
│   └── run_mae_pretraining.py     # entry point: python -m VideoMAE.run_mae_pretraining
├── SaVi/                          # owner's code (MIT)
│   ├── generate_simple_saliency_subset.py   # frames -> saliency PNGs
│   ├── precompute_saliency_patches.py       # saliency PNGs -> [T, H_p, W_p] .npy
│   ├── saliency_masking_generator.py        # the masking (per-group ratios, fixed K)
│   ├── saliency_io.py                       # saliency path/loading, error|random fallback
│   └── transform.py                         # dataset transform producing (video, mask)
├── scripts/                       # eval / reconstruction / list / plot utilities (argparse)
├── tests/                         # pytest, CPU, no dataset needed
├── site/                          # static project site
├── checkpoints/, datasets/        # (ignored by Git) local outputs and data
└── pyproject.toml, uv.lock        # uv project for the owner's code (tests, lint)
```

---

## 🎞️ Saliency-Aware Pretraining Overview

SaViMAE extends VideoMAE by using **saliency maps** to guide the masking strategy during self-supervised video reconstruction.

### Pipeline Summary

1. **Frame Extraction**  
   UCF101 videos are pre-extracted into folders of `img_XXXXX.jpg`.

2. **Saliency Map Generation**  
   A lightweight saliency method computes grayscale saliency for each frame:
   > saliency = | pixel – frame_mean | normalized to [0,1]

3. **Patch Saliency Computation**  
   Each saliency PNG → patch-level saliency via:
   - thresholding,
   - 16×16 patch grouping,
   - max-pooling within each patch.

   Produces `.npy` files of shape:  
   **[T, H_p, W_p]**

4. **SaVi Masking Generator**  
   During pretraining, salient and non-salient (background) tokens are masked at different ratios, so salient regions stay *more* visible:
   - salient: 0.85 mask ratio  
   - background: 0.95 mask ratio  

   Token saliency comes from the frames the clip actually samples: each tubelet is salient if any of its frames is. VideoMAE needs the same number of masked tokens for every clip in a batch, so the total `K` is fixed (default `0.95 * N`, i.e. the same visible-token budget as the earlier runs, override with `--savi_total_mask_ratio`) and split between the groups in proportion to `ratio * group_size`. Realised group ratios therefore match the targets exactly only when `K` equals the sum of the per-group targets; otherwise they shift together and salient stays below background. With about 23% salient tokens the defaults give roughly 0.87 salient / 0.97 background; with 50% salient the background group saturates at 1.0, so lower `--savi_total_mask_ratio` if your saliency is dense. Masks are reproducible given `--seed` and change every epoch.

5. **VideoMAE Reconstruction Pretraining**  
   Standard MAE reconstruction loss is applied:
   - Tubelet size 2  
   - Patch size 16  
   - 16-frame clips  
   - CPU-friendly configuration  

---

## 📂 Expected Dataset Structure

After preparing your data (either manually or via Slurm jobs), your directories look like:

UCF-101-frames/
ClassName/
v_ClassName_gXX_cXX/
img_00001.jpg
img_00002.jpg
...

UCF101-saliency-full/
ClassName/
v_ClassName_gXX_cXX/
img_00001.png
img_00002.png
...

UCF101-saliency-patches-full/
ClassName/
v_ClassName_gXX_cXX.npy


---

## 🚀 Usage

### 1. Generate simple saliency maps

```bash
python SaVi/generate_simple_saliency_subset.py \
    --frames_root path/to/UCF-101-frames \
    --out_root path/to/UCF101-saliency-full \
    --input_size 224
```
### 2. Generate patch-level saliency .npy files

```bash
python SaVi/precompute_saliency_patches.py \
    --sal_root path/to/UCF101-saliency-full \
    --out_root path/to/UCF101-saliency-patches-full \
    --input_size 224 \
    --patch_size 16 \
    --threshold 0.5
```
### 3. Pretrain SaViMAE

```bash
uv sync --extra train        # torchvision, timm, einops, decord, matplotlib
python -m VideoMAE.run_mae_pretraining \
    --model pretrain_videomae_base_patch16_224 \
    --data_path pretrain_list_ucf101_full.txt \
    --mask_type savi \
    --saliency_root path/to/UCF101-saliency-patches-full \
    --salient_mask_ratio 0.85 \
    --background_mask_ratio 0.95 \
    --num_frames 16 \
    --sampling_rate 4 \
    --input_size 224 \
    --batch_size 2 \
    --epochs 30 \
    --device cpu \
    --output_dir checkpoints/ucf_savi_full
```

`--nonsalient_mask_ratio` still works as an alias of `--background_mask_ratio`. `--saliency_root` is required and honoured. A missing or corrupt saliency file aborts the run by default; `--saliency_fallback random` masks that clip uniformly at random instead, logs a warning, and prints the number of such clips at the end of each epoch (also stored as `saliency_fallback_clips` in `log.txt`). The saliency grid must match the token grid (e.g. 14x14 for 224 px / 16 px patches), otherwise it errors rather than resizing.

The list files hold one `<frames_dir> <label>` per line; only the last two path components (`<class>/<video>`) are used to find the saliency file. `pretrain_list_ucf101_full.txt` is relative; `scripts/make_ucf101_full_list.py` regenerates it for your own layout.

### 4. Evaluate / visualise a checkpoint

```bash
python -m scripts.eval_savi_recon_loss --data_path LIST --saliency_root DIR --checkpoint CKPT
python -m scripts.dump_savi_reconstructions --data_path LIST --saliency_root DIR --checkpoint CKPT --out_dir OUT
python scripts/plot_savi_loss.py --log checkpoints/run/train_log.txt --out loss.png
```

---

## 🛠️ Development

```bash
uv sync
uv run pytest                                  # CPU, tiny synthetic tensors, no dataset
uv run ruff check . && uv run ruff format --check SaVi scripts tests site/tools
```

CI (`.github/workflows/ci.yml`) runs the same on every PR. Lint covers the owner's code only; `VideoMAE/` is upstream and excluded.

---

## 🐛 Masking bug fixed 2026-10-22 (earlier results used the old behaviour)

Before this fix the training path (`DataAugmentationForSaViMAE` in `VideoMAE/datasets.py`) did not implement the method described above:

- It set a total budget `K = 0.95 * N`, gave 0.85 of `K` to salient tokens (capped at how many exist) and filled the rest from background. When salient tokens are a minority, **all of them were masked**, the opposite of the intent. On a synthetic example with 21% salient tokens: salient 100% masked, background 93.6%; now 87% and 97%.
- The per-group generator in `SaVi/saliency_masking_generator.py` was correct in spirit but never used by training; there is now one implementation, used by training.
- The saliency directory was hardcoded to `datasets/UCF101-saliency-patches-subset`, ignoring `--saliency_root`.
- A missing saliency file silently treated the whole clip as salient, which is plain random 0.95 masking, with only a printed warning.
- Saliency was taken from the *first* few frames of the video, while clips sample frames uniformly over the whole video, so saliency and tokens were misaligned in time.

**Results reported before this date were produced with the old behaviour and have not been rerun.** Treat them as describing "budget-first masking that hides salient tokens", not as results for the method above.

Other things to know: list files `pretrain_list.txt` and `pretrain_list_subset.txt` contain absolute cluster paths including a NetID (left as committed); the unused tube/random `--mask_type` options were removed because that path never produced a valid mask.

---

## 📝 Acknowledgments

RGBD_Video_SOD authors - Junhao Lin, Lei Zhu, Jiaxing Shen, Huazhu Fu, Qing Zhang, Liansheng Wang @ [https://arxiv.org/abs/2406.12536]

VideoMAE authors - Zhan Tong, Yibing Song, Jue Wang, Limin Wang @ [https://arxiv.org/abs/2203.12602]

UCF101 dataset creators - [https://www.crcv.ucf.edu/data/UCF101.php]

Dartmouth HPC mantainers/discovery cluster

Prof. SouYoung Jin
