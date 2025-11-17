import os
from types import SimpleNamespace

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from VideoMAE.datasets import build_pretraining_dataset
from VideoMAE.modeling_pretrain import pretrain_videomae_base_patch16_224


def build_args():
    return SimpleNamespace(
        data_path="pretrain_list_subset.txt",
        mask_type="savi",
        saliency_root="datasets/UCF101-saliency-patches-subset",
        salient_mask_ratio=0.85,
        nonsalient_mask_ratio=0.95,
        num_frames=16,
        sampling_rate=4,
        input_size=224,
        window_size=16,
        tubelet_size=2,
        batch_size=2,
        num_workers=2,
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
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def reconstruct_videos(model, videos, bool_masked_pos, patch_size=16, normlize_target=True, device="cpu"):
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

    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

    # Unnormalize original to [0,1]
    unnorm_videos = videos * std + mean  # [B, C, T, H, W]

    # Patchify (same as in training)
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
    device = torch.device("cpu")
    args = build_args()
    loader = build_dataloader(args)

    model = load_model("checkpoints/ucf_savi_subset/checkpoint-0.pth", device)

    out_dir = "checkpoints/ucf_savi_subset/reconstructions"
    os.makedirs(out_dir, exist_ok=True)

    num_samples_to_save = 4
    saved = 0

    for batch_idx, (videos, bool_masked_pos) in enumerate(loader):
        if saved >= num_samples_to_save:
            break

        orig, recon, mask = reconstruct_videos(
            model, videos, bool_masked_pos, patch_size=16, normlize_target=True, device=device
        )

        B = orig.shape[0]
        for b in range(B):
            if saved >= num_samples_to_save:
                break
            out_path = os.path.join(out_dir, f"sample_{batch_idx}_{b}.png")
            save_recon_grid(orig[b], recon[b], mask[b].numpy(), out_path, max_frames=4)
            print(f"Saved {out_path}")
            saved += 1

    print(f"Saved {saved} reconstruction grids in {out_dir}")


if __name__ == "__main__":
    main()
