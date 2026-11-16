"""Average reconstruction MSE on masked tokens for a SaViMAE checkpoint (CPU).

Run from the repo root:
    python -m scripts.eval_savi_recon_loss --data_path LIST --saliency_root DIR --checkpoint CKPT
"""

import argparse
import math

import torch

from scripts.common import (
    add_data_args,
    build_dataloader,
    load_model,
    patchify_targets,
    unnormalize,
)


def eval_reconstruction(
    model, data_loader, device, patch_size=16, normlize_target=True, max_batches=None
):
    """
    Reproduces the label construction logic from engine_for_pretraining,
    but just computes average MSE on the masked tokens.
    """
    loss_fn = torch.nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for step, (videos, bool_masked_pos) in enumerate(data_loader):
            videos = videos.to(device, non_blocking=True)
            bool_masked_pos = (
                bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            )

            videos_patch = patchify_targets(
                unnormalize(videos), patch_size, normlize_target
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
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)

    loader = build_dataloader(args)
    print(f"Eval dataset size: {len(loader.dataset)} videos")
    model = load_model(args.checkpoint, device)

    avg_mse = eval_reconstruction(model, loader, device, max_batches=args.max_batches)
    print(f"\nAverage reconstruction MSE per masked token: {avg_mse:.6f}")
    skipped = loader.dataset.pop_fallback_count()
    if skipped:
        print(
            f"[saliency] {skipped} clips used random masking (missing/corrupt saliency)"
        )


if __name__ == "__main__":
    main()
