import os
import argparse
from PIL import Image
import numpy as np

def compute_simple_saliency(gray_arr: np.ndarray) -> np.ndarray:
    """
    gray_arr: H x W, float32 in [0, 1]
    Saliency = |I - mean(I)|, normalized to [0, 1].
    """
    mean_val = gray_arr.mean()
    sal = np.abs(gray_arr - mean_val)
    max_val = sal.max()
    if max_val > 0:
        sal = sal / max_val
    return sal

def process_frames(frames_root: str, out_root: str, input_size: int = 224):
    """
    Walk frames_root (class/video/frame.jpg), write saliency PNGs
    to out_root with mirrored structure.
    """
    frames_root = os.path.abspath(frames_root)
    out_root = os.path.abspath(out_root)

    num_images = 0
    for class_name in sorted(os.listdir(frames_root)):
        class_dir = os.path.join(frames_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for video_name in sorted(os.listdir(class_dir)):
            video_dir = os.path.join(class_dir, video_name)
            if not os.path.isdir(video_dir):
                continue

            # Mirror: out_root/class/video
            out_video_dir = os.path.join(out_root, class_name, video_name)
            os.makedirs(out_video_dir, exist_ok=True)

            frame_files = sorted(
                f for f in os.listdir(video_dir) if f.lower().endswith(".jpg")
            )

            for frame_file in frame_files:
                in_path = os.path.join(video_dir, frame_file)
                base, _ = os.path.splitext(frame_file)
                out_path = os.path.join(out_video_dir, base + ".png")

                # Skip if already exists (so you can resume)
                if os.path.exists(out_path):
                    continue

                try:
                    img = Image.open(in_path).convert("L")  # grayscale
                    if input_size is not None:
                        img = img.resize((input_size, input_size), Image.BILINEAR)

                    arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
                    sal = compute_simple_saliency(arr)              # [0,1]

                    sal_img = (sal * 255.0).astype(np.uint8)
                    out_img = Image.fromarray(sal_img, mode="L")
                    out_img.save(out_path)

                    num_images += 1
                    if num_images % 1000 == 0:
                        print(f"Processed {num_images} frames...")

                except Exception as e:
                    print(f"Error processing {in_path}: {e}")

    print(f"Done! Total saliency maps written: {num_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames_root",
        type=str,
        required=True,
        help="Root of subset frames (e.g., datasets/UCF-101-frames-subset)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root for saliency PNGs (e.g., datasets/UCF101-saliency-subset)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Resize frames to this size before computing saliency",
    )
    args = parser.parse_args()

    process_frames(args.frames_root, args.out_root, args.input_size)

