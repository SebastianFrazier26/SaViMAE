import math
import re
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from VideoMAE.datasets import build_pretraining_dataset
from VideoMAE.modeling_pretrain import pretrain_videomae_base_patch16_224


def build_args():
    """
    Minimal args namespace to reuse build_pretraining_dataset
    with your existing config.
    """
    return SimpleNamespace(
        # dataset / transforms
        data_path="pretrain_list_subset.txt",
        mask_type="savi",
        saliency_root="datasets/UCF101-saliency-patches-subset",
        salient_mask_ratio=0.85,
        nonsalient_mask_ratio=0.95,
        num_frames=16,
        sampling_rate=4,
        input_size=224,
        # not used, but some code may expect these
        window_size=16,
        tubelet_size=2,
        # batch size for this eval
        batch_size=2,
        num_workers=4,
    )


def build_dataloader(args):
    dataset = build_pretraining_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return loader


def load_model(checkpoint_path, device):
    model = pretrain_videomae_base_patch16_224()
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Handle both {'model': state_dict} and plain state_dict
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def eval_reconstruction(model, data_loader, device, patch_size=16, normlize_target=True, max_batches=None):
    """
    Reproduces the label construction logic from engine_for_pretraining,
    but just computes average MSE on the masked tokens.
    """
    loss_fn = torch.nn.MSELoss(reduction="sum")

    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for step, (videos, bool_masked_pos) in enumerate(data_loader):
            videos = videos.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

            # Unnormalize videos to [0,1]
            unnorm_videos = videos * std + mean  # [B, C, T, H, W]

            if normlize_target:
                videos_squeeze = rearrange(
                    unnorm_videos,
                    "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
                    p0=2,
                    p1=patch_size,
                    p2=patch_size,
                )
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)) / (
                    videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                )
                videos_patch = rearrange(videos_norm, "b n p c -> b n (p c)")
            else:
                videos_patch = rearrange(
                    unnorm_videos,
                    "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
                    p0=2,
                    p1=patch_size,
                    p2=patch_size,
                )

            B, N, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)  # [B, N_mask, C]

            outputs = model(videos, bool_masked_pos)  # same shape

            # Sum MSE over all masked tokens
            mse_sum = loss_fn(outputs, labels)
            num_tokens = labels.numel() / C  # masked tokens count

            total_loss += mse_sum.item()
            total_tokens += num_tokens

            if max_batches is not None and (step + 1) >= max_batches:
                break

    avg_mse_per_token = total_loss / total_tokens if total_tokens > 0 else math.nan
    return avg_mse_per_token


def main():
    device = torch.device("cpu")  # your env is CPU-only
    args = build_args()
    print("Building eval dataloader...")
    loader = build_dataloader(args)
    print(f"Eval dataset size: {len(loader.dataset)} videos")

    print("Loading model from checkpoints/ucf_savi_subset/checkpoint-0.pth")
    model = load_model("checkpoints/ucf_savi_subset/checkpoint-0.pth", device)

    print("Evaluating reconstruction MSE on masked tokens...")
    avg_mse = eval_reconstruction(model, loader, device, max_batches=None)
    print(f"\nAverage reconstruction MSE per masked token: {avg_mse:.6f}")


if __name__ == "__main__":
    main()

