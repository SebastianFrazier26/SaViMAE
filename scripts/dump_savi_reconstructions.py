"""Save original / reconstruction / mask grids for a SaViMAE checkpoint (CPU).

Run from the repo root:
    python -m scripts.dump_savi_reconstructions --data_path LIST --saliency_root DIR \
        --checkpoint CKPT --out_dir OUT
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from einops import rearrange

from scripts.common import (
    add_data_args,
    build_dataloader,
    load_model,
    patchify_targets,
    unnormalize,
)


def reconstruct_videos(
    model, videos, bool_masked_pos, patch_size=16, normlize_target=True, device="cpu"
):
    """
    Reconstruct videos from model outputs by filling masked tokens with predictions
    and unpatchifying back to video space.

    Returns:
        orig_videos:  [B, C, T, H, W] in [0,1]
        recon_videos: [B, C, T, H, W] (roughly in [0,1])
        mask:         [B, N_tokens] boolean
    """
    device = torch.device(device)
    videos = videos.to(device)
    bool_masked_pos = bool_masked_pos.to(device).flatten(1).to(torch.bool)

    unnorm_videos = unnormalize(videos)  # [B, C, T, H, W]
    videos_patch = patchify_targets(unnorm_videos, patch_size, normlize_target)

    B, N, C = videos_patch.shape

    with torch.no_grad():
        pred = model(videos, bool_masked_pos)  # [B, N_mask, C]

    # Fill masked tokens with predictions (shape-aware)
    full_tokens = videos_patch.clone()  # [B, N, C]
    full_tokens[bool_masked_pos] = pred.reshape(-1, C)  # [B*N_mask, C]

    # ---- Unpatchify back to video space ----
    # Try using model.unpatchify if available (VideoMAE-style)
    if hasattr(model, "unpatchify"):
        # model.unpatchify expects [B, N, C] where C is patch dim
        vids_rec = model.unpatchify(full_tokens)
        # Shape should be [B, 3, T, H, W] or similar; we assume that's what we want.
    else:
        # Fallback: manual inverse of the rearrange above
        p0, p1, p2 = 2, patch_size, patch_size
        c = 3
        patches = full_tokens.view(B, N, p0 * p1 * p2, c)

        # Recover T, H, W from original
        _, _, T, H, W = unnorm_videos.shape
        H_p = H // patch_size
        W_p = W // patch_size
        t_tok = T // p0

        vids_rec = rearrange(
            patches,
            "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)",
            t=t_tok,
            h=H_p,
            w=W_p,
            p0=p0,
            p1=p1,
            p2=p2,
        )

    # Clip for visualization
    vids_rec = vids_rec.clamp(0.0, 1.0)

    return unnorm_videos.cpu(), vids_rec.cpu(), bool_masked_pos.cpu()


def save_recon_grid(orig, recon, mask, out_path, max_frames=4):
    """
    Save a simple grid: rows = [original, reconstructed, mask] for a single sample.
    """
    C, T, H, W = orig.shape
    T_show = min(T, max_frames)

    fig, axes = plt.subplots(3, T_show, figsize=(3 * T_show, 6))

    for t in range(T_show):
        # Original
        ax = axes[0, t] if T_show > 1 else axes[0]
        frame = orig[:, t].permute(1, 2, 0).numpy()
        ax.imshow(frame)
        ax.axis("off")
        if t == 0:
            ax.set_ylabel("Original")

        # Reconstructed
        ax = axes[1, t] if T_show > 1 else axes[1]
        frame_rec = recon[:, t].permute(1, 2, 0).numpy()
        ax.imshow(frame_rec)
        ax.axis("off")
        if t == 0:
            ax.set_ylabel("Recon")

        # Mask visualization (simple 1D barcode of tokens)
        ax = axes[2, t] if T_show > 1 else axes[2]
        ax.imshow(mask.reshape(1, -1), cmap="gray", aspect="auto")
        ax.axis("off")
        if t == 0:
            ax.set_ylabel("Mask tokens")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device)

    loader = build_dataloader(args)
    model = load_model(args.checkpoint, device)
    os.makedirs(args.out_dir, exist_ok=True)

    saved = 0
    for batch_idx, (videos, bool_masked_pos) in enumerate(loader):
        if saved >= args.num_samples:
            break
        orig, recon, mask = reconstruct_videos(
            model,
            videos,
            bool_masked_pos,
            patch_size=16,
            normlize_target=True,
            device=device,
        )
        for b in range(orig.shape[0]):
            if saved >= args.num_samples:
                break
            out_path = os.path.join(args.out_dir, f"sample_{batch_idx}_{b}.png")
            save_recon_grid(orig[b], recon[b], mask[b].numpy(), out_path, max_frames=4)
            print(f"Saved {out_path}")
            saved += 1

    print(f"Saved {saved} reconstruction grids in {args.out_dir}")


if __name__ == "__main__":
    main()
