# SaViMAE — Saliency-Aware VideoMAE Pretraining

SaViMAE is a saliency-guided extension of VideoMAE designed for efficient self-supervised video representation learning.  
This repository contains:

- A simplified VideoMAE pretraining pipeline  
- Frame-based UCF101 data loading  
- Saliency map generation from raw frames  
- Patch-level saliency computation  
- A custom masking generator for saliency-aware training  
- CPU/HPC-friendly scripts + Slurm support  

This project was built for academic experimentation and course work, emphasizing:
- clear reproducibility,
- no reliance on external RGB-D models,
- compatibility with CPU-only environments (e.g., Dartmouth Polaris),
- and a full saliency → patch → pretraining workflow.

---

## 📦 Repository Structure

SaViMAE/
│
├── VideoMAE/ # VideoMAE model, datasets, transforms, utils
├── SaVi/ # Saliency + patch computation, masking generator
│ ├── generate_simple_saliency_subset.py
│ ├── precompute_saliency_patches.py
│ └── saliency_masking_generator.py
│
├── scripts/ # Reconstruction / evaluation utilities
│ └── dump_savi_reconstructions.py
│
├── checkpoints/ # (ignored by Git) training checkpoints, logs
├── datasets/ # (ignored by Git) local symlinks → scratch data
├── slurm_logs/ # (ignored by Git) HPC logs
│
├── pretrain_list_ucf101_full.txt
├── run_mae_pretraining.py
├── README.md
└── .gitignore


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
   During pretraining, salient > non-salient patches are masked at different ratios:
   - salient: 0.85 mask ratio  
   - non-salient: 0.95 mask ratio  

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
### 3. 3. Pretrain SaViMAE

```bash
python -m VideoMAE.run_mae_pretraining \
    --model pretrain_videomae_base_patch16_224 \
    --data_path pretrain_list_ucf101_full.txt \
    --mask_type savi \
    --saliency_root path/to/UCF101-saliency-patches-full \
    --salient_mask_ratio 0.85 \
    --nonsalient_mask_ratio 0.95 \
    --num_frames 16 \
    --sampling_rate 4 \
    --input_size 224 \
    --batch_size 2 \
    --epochs 30 \
    --device cpu \
    --output_dir checkpoints/ucf_savi_full
```
## 📝 Acknowledgments

RGBD_Video_SOD authors - Junhao Lin, Lei Zhu, Jiaxing Shen, Huazhu Fu, Qing Zhang, Liansheng Wang @ [https://arxiv.org/abs/2406.12536]

RGBD Video SOD authors - Zhan Tong, Yibing Song, Jue Wang, Limin Wang @ [https://arxiv.org/abs/2203.12602]

UCF101 dataset creators - [https://www.crcv.ucf.edu/data/UCF101.php]

Dartmouth HPC mantainers/discovery cluster

Prof. SouYoung Jin
