"""Compare the last logged train top-1 of a baseline and a SaVi run.

python SaVi/compare_results.py --baseline_log A/log.txt --savi_log B/log.txt --num_videos N
"""

import argparse
import json


def read_last_log(path: str) -> tuple[float, float | None]:
    with open(path) as f:
        last = json.loads(f.readlines()[-1])
    return last["train_top1"], last.get("train_top5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_log", required=True)
    parser.add_argument("--savi_log", required=True)
    parser.add_argument(
        "--num_videos",
        type=int,
        required=True,
        help="evaluation set size, for error counts",
    )
    args = parser.parse_args()

    base_top1, _ = read_last_log(args.baseline_log)
    savi_top1, _ = read_last_log(args.savi_log)
    print(f"Baseline Top-1: {base_top1:.2f}")
    print(f"SaVi Top-1:     {savi_top1:.2f}")
    print(f"Baseline errors (approx): {args.num_videos * (1 - base_top1 / 100.0):.1f}")
    print(f"SaVi errors (approx):     {args.num_videos * (1 - savi_top1 / 100.0):.1f}")


if __name__ == "__main__":
    main()
