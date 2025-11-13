# Sebastian Frazier
# Main file for training and testing

# SaVi/compare_results.py

import json
import os

def read_last_log(path):
    with open(path, "r") as f:
        lines = f.readlines()
    last = json.loads(lines[-1])
    return last['train_top1'], last.get('train_top5', None)

def main():
    base_log = "../checkpoints/baseline_k400/log.txt"
    savi_log = "../checkpoints/savi_k400/log.txt"

    base_top1, _ = read_last_log(base_log)
    savi_top1, _ = read_last_log(savi_log)

    print(f"Baseline Top-1: {base_top1:.2f}")
    print(f"SaVi Top-1:     {savi_top1:.2f}")

    # If you know dataset size N, error counts:
    N = 40000  # example
    base_errors = N * (1 - base_top1 / 100.0)
    savi_errors = N * (1 - savi_top1 / 100.0)

    print(f"Baseline errors (approx): {base_errors:.1f}")
    print(f"SaVi errors (approx):     {savi_errors:.1f}")

if __name__ == "__main__":
    main()
