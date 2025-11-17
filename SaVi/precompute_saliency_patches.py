import os
import argparse
from typing import List

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback if tqdm is not available


def load_saliency_frame(path: str, input_size: int) -> np.ndarray:
    """
    Load a single saliency frame (PNG), resize to (input_size, input_size),
    convert to grayscale, and return as float32 array in [0, 1].
    """
    img = Image.open(path).convert("L")
    img = img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def frame_to_patch_mask(frame: np.ndarray, patch_size: int, threshold: float) -> np.ndarray:
    """
    Convert a single saliency frame (H, W) in [0, 1] to a patch-level saliency mask.

    - threshold to binary (salient vs non-salient)
    - reshape into (H_p, patch_size, W_p, patch_size)
    - use max within each patch to decide if the patch is salient

    Returns:
        patch_mask: (H_p, W_p) float32 array in {0.0, 1.0}
    """
    H, W = frame.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Frame size ({H}, {W}) must be divisible by patch_size={patch_size}"

    binary = (frame >= threshold).astype(np.float32)

    Hp = H // patch_size
    Wp = W // patch_size

    # reshape to [Hp, patch_size, Wp, patch_size]
    reshaped = binary.reshape(Hp, patch_size, Wp, patch_size)

    # max pool within each patch
    patch_mask = reshaped.max(axis=(1, 3))
    return patch_mask  # (Hp, Wp)


def process_video(sal_vid_dir: str,
                  out_path: str,
                  input_size: int,
                  patch_size: int,
                  threshold: float) -> None:
    """
    Process all saliency frames in a single video directory:

    - load all saliency PNGs in sorted order
    - convert each to a patch-level saliency mask
    - stack to shape [T, H_p, W_p]
    - save as .npy at out_path

    If there are no valid frames, skip gracefully.
    """
    frames = sorted(
        f for f in os.listdir(sal_vid_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not frames:
        print(f"[WARN] No saliency frames found in {sal_vid_dir}, skipping.")
        return

    patch_masks: List[np.ndarray] = []

    for fname in frames:
        fpath = os.path.join(sal_vid_dir, fname)
        try:
            frame = load_saliency_frame(fpath, input_size)
        except Exception as e:
            print(f"[WARN] Failed to load frame {fpath}: {e}")
            continue

        try:
            patch_mask = frame_to_patch_mask(frame, patch_size, threshold)
        except AssertionError as e:
            print(f"[WARN] Skipping frame {fpath} due to size mismatch: {e}")
            continue

        patch_masks.append(patch_mask)

    if len(patch_masks) == 0:
        print(f"[WARN] No valid patch masks for video {sal_vid_dir}, skipping.")
        return

    patch_masks = np.stack(patch_masks, axis=0)  # [T, H_p, W_p]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, patch_masks.astype(np.float32))
    # print(f"[INFO] Saved patch masks to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sal_root",
        type=str,
        required=True,
        help="Root directory of saliency maps, e.g. datasets/UCF101-saliency-subset"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output directory for patch saliency npy files"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input frame size (H, W) used for saliency + patches"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size (pixels) for MAE patches"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold in [0,1] to decide saliency vs non-saliency per pixel"
    )

    args = parser.parse_args()

    sal_root = args.sal_root
    out_root = args.out_root

    if not os.path.isdir(sal_root):
        raise RuntimeError(f"Saliency root directory not found: {sal_root}")

    os.makedirs(out_root, exist_ok=True)

    classes = sorted(
        d for d in os.listdir(sal_root)
        if os.path.isdir(os.path.join(sal_root, d))
    )

    print(f"[INFO] Found {len(classes)} classes in saliency root: {sal_root}")
    print(f"[INFO] Saving patch saliency to: {out_root}")
    print(f"[INFO] input_size={args.input_size}, patch_size={args.patch_size}, threshold={args.threshold}")

    for cls in tqdm(classes, desc="Classes"):
        cls_sal_dir = os.path.join(sal_root, cls)
        videos = sorted(
            d for d in os.listdir(cls_sal_dir)
            if os.path.isdir(os.path.join(cls_sal_dir, d))
        )

        for vid in tqdm(videos, desc=f"{cls}", leave=False):
            sal_vid_dir = os.path.join(cls_sal_dir, vid)
            out_path = os.path.join(out_root, cls, vid + ".npy")

            # Skip if already computed
            if os.path.exists(out_path):
                continue

            process_video(
                sal_vid_dir=sal_vid_dir,
                out_path=out_path,
                input_size=args.input_size,
                patch_size=args.patch_size,
                threshold=args.threshold,
            )

    print("[INFO] Done precomputing saliency patches.")


if __name__ == "__main__":
    main()

