import re
import matplotlib.pyplot as plt

LOG_PATH = "checkpoints/ucf_savi_subset/train_log.txt"

step_ids = []
losses = []

step = 0

with open(LOG_PATH, "r") as f:
    for line in f:
        if "Epoch:" in line and "loss:" in line:
            # Example fragment: "loss: 1.3768 (1.3768)"
            m = re.search(r"loss:\s*([0-9.]+)", line)
            if m:
                loss_val = float(m.group(1))
                losses.append(loss_val)
                step_ids.append(step)
                step += 1

print(f"Parsed {len(losses)} loss entries")

plt.figure()
plt.plot(step_ids, losses, marker="", linestyle="-")
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("SaViMAE pretraining loss (UCF-101 subset)")
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/ucf_savi_subset/loss_curve.png")
print("Saved checkpoints/ucf_savi_subset/loss_curve.png")

