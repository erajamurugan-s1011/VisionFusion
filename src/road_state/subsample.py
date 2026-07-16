import os
import random
import shutil

SOURCE_ROOT = "../../data/bdd100k_full"
DEST_ROOT = "../../data/bdd100k_subset"

# Target counts per split, kept small and free-GPU-friendly
SPLIT_TARGETS = {
    "train": 3000,
    "valid": 600,
    "test": 400,
}

random.seed(42)

for split, target_count in SPLIT_TARGETS.items():
    src_images = os.path.join(SOURCE_ROOT, split, "images")
    src_labels = os.path.join(SOURCE_ROOT, split, "labels")
    dst_images = os.path.join(DEST_ROOT, split, "images")
    dst_labels = os.path.join(DEST_ROOT, split, "labels")

    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    all_files = [f for f in os.listdir(src_images) if f.lower().endswith(".jpg")]
    random.shuffle(all_files)
    selected = all_files[:target_count]

    for fname in selected:
        base = os.path.splitext(fname)[0]
        shutil.copy2(os.path.join(src_images, fname), os.path.join(dst_images, fname))
        label_name = base + ".txt"
        src_label_path = os.path.join(src_labels, label_name)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, os.path.join(dst_labels, label_name))

    print(f"{split}: copied {len(selected)} images (target {target_count})")

print("Subsampling complete:", DEST_ROOT)