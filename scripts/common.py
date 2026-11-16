"""Helpers shared by the evaluation / visualisation scripts."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader

from VideoMAE.datasets import build_pretraining_dataset
from VideoMAE.modeling_pretrain import pretrain_videomae_base_patch16_224


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data_path", required=True, help="list file: '<frames_dir> <label>' per line"
    )
    parser.add_argument(
        "--saliency_root",
        required=True,
        help="root of <class>/<video>.npy patch saliency",
    )
    parser.add_argument(
        "--saliency_fallback", choices=["error", "random"], default="error"
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pth to load")
    parser.add_argument("--salient_mask_ratio", type=float, default=0.85)
    parser.add_argument(
        "--background_mask_ratio",
        "--nonsalient_mask_ratio",
        dest="background_mask_ratio",
        type=float,
        default=0.95,
    )
    parser.add_argument("--savi_total_mask_ratio", type=float, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--seed", type=int, default=0, help="mask seed (fixed so evals are repeatable)"
    )
    parser.add_argument("--device", default="cpu")


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = build_pretraining_dataset(SimpleNamespace(mask_type="savi", **vars(args)))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def load_model(checkpoint_path: str, device: torch.device):
    model = pretrain_videomae_base_patch16_224()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # strict=False is from the original scripts; mismatches are printed rather than hidden.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[load_model] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )
    return model.to(device).eval()


def unnormalize(videos: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN, device=videos.device)[
        None, :, None, None, None
    ]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD, device=videos.device)[
        None, :, None, None, None
    ]
    return videos * std + mean


def patchify_targets(
    unnorm_videos: torch.Tensor, patch_size: int, normlize_target: bool
) -> torch.Tensor:
    """[B, C, T, H, W] in [0,1] -> [B, N, p0*p1*p2*C]; mirrors engine_for_pretraining."""
    if normlize_target:
        squeezed = rearrange(
            unnorm_videos,
            "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
            p0=2,
            p1=patch_size,
            p2=patch_size,
        )
        normed = (squeezed - squeezed.mean(dim=-2, keepdim=True)) / (
            squeezed.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        )
        return rearrange(normed, "b n p c -> b n (p c)")
    return rearrange(
        unnorm_videos,
        "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
        p0=2,
        p1=patch_size,
        p2=patch_size,
    )
