# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pillow>=10"]
# ///
"""Render site/assets/masking.svg: the SaViMAE masking scheme on a synthetic frame.

Illustration only. The frame is generated here, not taken from UCF101, and the
percentages printed in the figure describe this one synthetic frame, not results.

Run from anywhere:  uv run site/tools/make_masking_figure.py
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "site" / "assets" / "masking.svg"

# Import the project's real saliency/patch/mask code so the figure tracks main.
sys.path.insert(0, str(REPO / "SaVi"))
from generate_simple_saliency_subset import compute_simple_saliency
from precompute_saliency_patches import frame_to_patch_mask
from saliency_masking_generator import SaliencyMaskingGenerator

SIZE = 224
PATCH = 16
GRID = SIZE // PATCH
TUBELETS = 8  # 16 frames / tubelet size 2
SALIENT_RATIO = 0.85
NONSALIENT_RATIO = 0.95
SEED = 7


def synthetic_frame() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    img = 0.38 + 0.14 * (xx / SIZE) + 0.03 * rng.standard_normal((SIZE, SIZE))
    body = ((xx - 92) / 26) ** 2 + ((yy - 128) / 58) ** 2 <= 1
    head = ((xx - 92) / 17) ** 2 + ((yy - 54) / 17) ** 2 <= 1
    ball = ((xx - 168) / 15) ** 2 + ((yy - 150) / 15) ** 2 <= 1
    img[body | head] = 0.95
    img[ball] = 0.04
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def training_path_mask(
    flat_saliency: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Mirror of DataAugmentationForSaViMAE.__call__ in VideoMAE/datasets.py.

    Re-stated rather than imported because that module pulls in torch/torchvision;
    keep in sync if the budget logic there changes.
    """
    total = flat_saliency.shape[0]
    salient_idx = np.where(flat_saliency == 1)[0]
    nonsalient_idx = np.where(flat_saliency == 0)[0]
    k = max(1, min(total - 1, int(NONSALIENT_RATIO * total)))
    k_sal = int(SALIENT_RATIO * k)
    k_nonsal = k - k_sal
    chosen = []
    if len(salient_idx) > 0 and k_sal > 0:
        k_sal = min(k_sal, len(salient_idx))
        chosen.append(rng.choice(salient_idx, size=k_sal, replace=False))
    if len(nonsalient_idx) > 0 and k_nonsal > 0:
        k_nonsal = min(k_nonsal, len(nonsalient_idx))
        chosen.append(rng.choice(nonsalient_idx, size=k_nonsal, replace=False))
    picked = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)
    if picked.shape[0] < k:
        remaining = np.setdiff1d(np.arange(total), picked, assume_unique=True)
        extra = min(k - picked.shape[0], remaining.shape[0])
        picked = np.concatenate(
            [picked, rng.choice(remaining, size=extra, replace=False)]
        )
    mask = np.zeros(total, dtype=np.int64)
    mask[picked] = 1
    return mask


def png_data_uri(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(
        buf, format="PNG", optimize=True
    )
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def group_stats(mask2d: np.ndarray, sal2d: np.ndarray) -> tuple[str, str]:
    sal = sal2d == 1

    def pct(sel: np.ndarray) -> str:
        n = int(sel.sum())
        return (
            f"{int(mask2d[sel].sum())}/{n} ({100 * mask2d[sel].mean():.0f}%)"
            if n
            else "0/0"
        )

    return pct(sal), pct(~sal)


def panel(
    x: float, y: float, size: float, title: str, lines: list[str], body: str
) -> str:
    text = [f'<text class="t" x="{x}" y="{y - 10}">{title}</text>']
    for i, line in enumerate(lines):
        text.append(
            f'<text class="s" x="{x}" y="{y + size + 18 + 15 * i}">{line}</text>'
        )
    return f'<g transform="translate({x} {y})">{body}</g>' + "".join(text)


def grid_rects(cell: float, pred, cls: str) -> str:
    out = []
    for i in range(GRID):
        for j in range(GRID):
            if pred(i, j):
                out.append(
                    f'<rect class="{cls}" x="{j * cell:.2f}" y="{i * cell:.2f}" '
                    f'width="{cell:.2f}" height="{cell:.2f}"/>'
                )
    return "".join(out)


def main() -> None:
    frame = synthetic_frame()
    saliency = compute_simple_saliency(frame)
    patches = frame_to_patch_mask(saliency, PATCH, threshold=0.5)
    volume = np.repeat(patches[None], TUBELETS, axis=0)

    np.random.seed(SEED)  # SaliencyMaskingGenerator draws from the global RNG
    gen = SaliencyMaskingGenerator(
        (TUBELETS, GRID, GRID), SALIENT_RATIO, NONSALIENT_RATIO, volume
    )
    mask_a = gen().reshape(TUBELETS, GRID, GRID)[0]
    sal_flat = (volume >= 0.5).astype(np.int32).reshape(-1)
    mask_b = training_path_mask(sal_flat, np.random.default_rng(SEED)).reshape(
        TUBELETS, GRID, GRID
    )[0]
    sal2d = (patches >= 0.5).astype(np.int32)

    s = 200.0
    cell = s / GRID
    gap_x, gap_y = 40.0, 110.0
    xs = [20.0, 20.0 + s + gap_x, 20.0 + 2 * (s + gap_x)]
    ys = [50.0, 50.0 + s + gap_y]
    width = xs[2] + s + 30
    height = ys[1] + s + 80

    frame_img = f'<image href="{png_data_uri(frame)}" width="{s}" height="{s}"/>'
    sal_img = f'<image href="{png_data_uri(saliency)}" width="{s}" height="{s}"/>'
    outline = grid_rects(cell, lambda i, j: sal2d[i, j] == 1, "sal")

    def masked_view(mask2d: np.ndarray) -> str:
        return (
            frame_img + grid_rects(cell, lambda i, j: mask2d[i, j] == 1, "m") + outline
        )

    a_sal, a_non = group_stats(mask_a, sal2d)
    b_sal, b_non = group_stats(mask_b, sal2d)
    n_sal = int(sal2d.sum())

    parts = [
        panel(
            xs[0],
            ys[0],
            s,
            "1 · synthetic frame (224×224)",
            ["made up in this script —", "not a UCF101 frame"],
            frame_img,
        ),
        panel(
            xs[1],
            ys[0],
            s,
            "2 · saliency |I − mean(I)|",
            ["compute_simple_saliency()", "SaVi/generate_simple_saliency_subset.py"],
            sal_img,
        ),
        panel(
            xs[2],
            ys[0],
            s,
            "3 · patch saliency (14×14)",
            [
                "threshold 0.5, max-pool 16×16",
                f"{n_sal}/196 patches salient",
                "frame_to_patch_mask()",
            ],
            grid_rects(cell, lambda i, j: sal2d[i, j] == 1, "fill")
            + grid_rects(cell, lambda i, j: True, "cellline"),
        ),
        panel(
            xs[0],
            ys[1],
            s,
            "4 · SaliencyMaskingGenerator",
            [
                f"salient masked {a_sal}",
                f"non-salient masked {a_non}",
                "per-group ratios 0.85 / 0.95",
            ],
            masked_view(mask_a),
        ),
        panel(
            xs[1],
            ys[1],
            s,
            "5 · training path (datasets.py)",
            [
                f"salient masked {b_sal}",
                f"non-salient masked {b_non}",
                "budget K = 0.95·N, 85% of K to salient",
            ],
            masked_view(mask_b),
        ),
    ]

    lx, ly = xs[2], ys[1]
    legend = (
        f'<text class="t" x="{lx}" y="{ly - 10}">legend</text>'
        f'<rect class="m" x="{lx}" y="{ly}" width="18" height="18"/><text class="s" x="{lx + 26}" y="{ly + 14}">masked token (hidden)</text>'
        f'<rect class="sal" x="{lx}" y="{ly + 28}" width="18" height="18"/><text class="s" x="{lx + 26}" y="{ly + 42}">salient patch outline</text>'
        f'<rect class="fill" x="{lx}" y="{ly + 56}" width="18" height="18"/><text class="s" x="{lx + 26}" y="{ly + 70}">salient patch</text>'
        f'<text class="s" x="{lx}" y="{ly + 104}">Shown: first of {TUBELETS} tubelet slices;</text>'
        f'<text class="s" x="{lx}" y="{ly + 119}">masks drawn with seed {SEED}.</text>'
        f'<text class="s" x="{lx}" y="{ly + 144}">Illustrative only — no metrics.</text>'
    )

    style = (
        "<style>"
        ".bg{fill:#ffffff}.t{font:600 13px system-ui,sans-serif;fill:#1b1f24}"
        ".s{font:11.5px system-ui,sans-serif;fill:#4b5563}"
        ".m{fill:#6b7280;fill-opacity:.88}.sal{fill:none;stroke:#d9480f;stroke-width:1.6}"
        ".fill{fill:#d9480f;fill-opacity:.85}.cellline{fill:none;stroke:#9ca3af;stroke-width:.4}"
        "@media (prefers-color-scheme: dark){.bg{fill:#14171c}.t{fill:#e6e8eb}.s{fill:#a3acb9}"
        ".m{fill:#2b3038}.sal{stroke:#ff8a4c}.fill{fill:#ff8a4c}.cellline{stroke:#4b5563}}"
        "</style>"
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-labelledby="ttl">'
        f'<title id="ttl">Illustrative SaViMAE masking on a synthetic frame (not a result)</title>'
        f'{style}<rect class="bg" width="100%" height="100%"/>'
        f'<text class="t" x="20" y="22">ILLUSTRATIVE — generated by site/tools/make_masking_figure.py from the repo code</text>'
        + "".join(parts)
        + legend
        + "</svg>\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(REPO)}  salient patches {n_sal}/196  "
        f"A: sal {a_sal} non {a_non}  B: sal {b_sal} non {b_non}"
    )


if __name__ == "__main__":
    main()
