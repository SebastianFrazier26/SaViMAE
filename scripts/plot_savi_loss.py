"""Plot the training loss from a VideoMAE train_log.txt.

python scripts/plot_savi_loss.py --log checkpoints/run/train_log.txt --out loss_curve.png
"""

import argparse
import re

import matplotlib.pyplot as plt

LOSS_RE = re.compile(r"loss:\s*([0-9.]+)")


def parse_losses(log_path: str) -> list[float]:
    losses = []
    with open(log_path) as f:
        for line in f:
            if "Epoch:" in line and "loss:" in line:
                m = LOSS_RE.search(line)
                if m:
                    losses.append(float(m.group(1)))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="SaViMAE pretraining loss")
    args = parser.parse_args()

    losses = parse_losses(args.log)
    print(f"Parsed {len(losses)} loss entries")
    plt.figure()
    plt.plot(range(len(losses)), losses, linestyle="-")
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title(args.title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
