import os
import random
import shutil

# ======================
# CONFIG
# ======================
source_dir = "dataset_all"   # folder asli (semua data)
output_dir = "dataset"       # hasil split

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# ======================
# PROSES
# ======================
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images   = images[train_end:val_end]
    test_images  = images[val_end:]

    # buat folder
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    # copy file
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(output_dir, "train", class_name, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(output_dir, "val", class_name, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(output_dir, "test", class_name, img)
        )

    print(f"✔ {class_name}: {len(train_images)} train | {len(val_images)} val | {len(test_images)} test")

print("\n🔥 Dataset split selesai!")