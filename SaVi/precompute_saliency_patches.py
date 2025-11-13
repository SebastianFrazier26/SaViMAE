# SaVi/precompute_saliency_patches.py

import os
import glob
import numpy as np
from PIL import Image

import argparse

def binarize_saliency_map(img: Image.Image, threshold: float = 0.5):
    """Convert saliency PNG into binary mask [H, W] with values in {0, 1}."""
    arr = np.array(img).astype("float32")
    # If saliency is single-channel [H, W], normalize to [0,1]
    if arr.ndim == 2:
        arr_norm = arr / (arr.max() + 1e-8)
    else:
        # If RGB, just use one channel / average
        arr_norm = arr.mean(axis=-1) / (arr.max() + 1e-8)
    return (arr_norm >= threshold).astype("float32")


def compute_patch_saliency(mask_hw, input_size=224, patch_size=16):
    """
    mask_hw: [H, W] at image resolution (e.g. 224x224).
    Returns patch-level mask [H_p, W_p] where each patch is salient if mean > 0.5.
    """
    H, W = mask_hw.shape
    assert H == input_size and W == input_size, "Resize saliency to input_size first"

    H_p = input_size // patch_size
    W_p = input_size // patch_size

    patch_mask = np.zeros((H_p, W_p), dtype=np.float32)
    for i in range(H_p):
        for j in range(W_p):
            h0 = i * patch_size
            h1 = (i + 1) * patch_size
            w0 = j * patch_size
            w1 = (j + 1) * patch_size
            patch = mask_hw[h0:h1, w0:w1]
            patch_mask[i, j] = 1.0 if patch.mean() >= 0.5 else 0.0

    return patch_mask


def process_video(video_id, sal_root, out_root, num_frames=None,
                  input_size=224, patch_size=16, threshold=0.5):
    """
    video_id: e.g. 'video_0001'
    sal_root: e.g. 'RGBD_Video_SOD/vidsod_100/saliency'
    out_root: directory to save patch-level npy
    """
    video_dir = os.path.join(sal_root, video_id)
    frame_paths = sorted(glob.glob(os.path.join(video_dir, "*.png")))
    if num_frames is not None:
        frame_paths = frame_paths[:num_frames]

    patch_masks = []
    for fp in frame_paths:
        img = Image.open(fp).convert("L")  # 1-channel
        img = img.resize((input_size, input_size), Image.BILINEAR)
        bin_mask = binarize_saliency_map(img, threshold=threshold)
        patch_mask = compute_patch_saliency(bin_mask, input_size=input_size,
                                            patch_size=patch_size)
        patch_masks.append(patch_mask)

    patch_masks = np.stack(patch_masks, axis=0)  # [T, H_p, W_p]

    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"{video_id}.npy")
    np.save(out_path, patch_masks)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sal_root", type=str, required=True,
                        help="Root folder of frame-level saliency maps (per video subdir)")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Output folder for patch-level saliency .npy")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Optional: limit frames per video")
    args = parser.parse_args()

    video_ids = sorted(os.listdir(args.sal_root))
    for vid in video_ids:
        if not os.path.isdir(os.path.join(args.sal_root, vid)):
            continue
        process_video(
            video_id=vid,
            sal_root=args.sal_root,
            out_root=args.out_root,
            num_frames=args.num_frames,
            input_size=args.input_size,
            patch_size=args.patch_size,
            threshold=args.threshold,
        )

if __name__ == "__main__":
    main()