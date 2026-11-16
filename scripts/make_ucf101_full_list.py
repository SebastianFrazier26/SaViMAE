"""Write a '<frames_dir> <label>' list for a UCF-101 frames tree.

Labels are the alphabetical class index, matching the subset list.
    python scripts/make_ucf101_full_list.py --frames_root datasets/UCF-101-frames --out list.txt
"""

import argparse
from pathlib import Path


def build_list(frames_root: Path) -> list[str]:
    classes = sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    lines = []
    for label, cls in enumerate(classes):
        for vid in sorted(d for d in (frames_root / cls).iterdir() if d.is_dir()):
            lines.append(f"{frames_root / cls / vid.name} {label}\n")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lines = build_list(args.frames_root)
    args.out.write_text("".join(lines))
    print(f"Wrote {len(lines)} videos to {args.out}")


if __name__ == "__main__":
    main()
