import os

FRAMES_ROOT = "datasets/UCF-101-frames"
OUT_LIST = "pretrain_list_ucf101_full.txt"

# Use the same class → label mapping as in the subset:
# sort class names alphabetically and assign 0..N-1
classes = sorted(
    d for d in os.listdir(FRAMES_ROOT)
    if os.path.isdir(os.path.join(FRAMES_ROOT, d))
)
class_to_label = {cls: i for i, cls in enumerate(classes)}

print("Found classes:")
for cls, lbl in class_to_label.items():
    print(f"  {lbl:2d}  {cls}")

lines = []

for cls in classes:
    label = class_to_label[cls]
    cls_dir = os.path.join(FRAMES_ROOT, cls)
    for vid in sorted(os.listdir(cls_dir)):
        vid_dir = os.path.join(cls_dir, vid)
        if not os.path.isdir(vid_dir):
            continue
        # line format: "path/to/video_dir label"
        rel_path = os.path.join(FRAMES_ROOT, cls, vid)
        lines.append(f"{rel_path} {label}\n")

print(f"\nTotal videos: {len(lines)}")
with open(OUT_LIST, "w") as f:
    f.writelines(lines)

print(f"Wrote list to {OUT_LIST}")

