import random
import shutil
from pathlib import Path

# HART KODIERTE PFADANGABEN
SOURCE_DIR = Path(r"./datasets/jpg")
TARGET_DIR = Path(r"./datasets/yolo_training_filter")

# Verhältnis Training/Validierung
TRAIN_RATIO = 0.8

def prepare_dirs(base_dir):
    for split in ['train', 'val']:
        for sub in ['images', 'labels']:
            path = base_dir / split / sub
            path.mkdir(parents=True, exist_ok=True)

def get_image_label_pairs(source_root):
    allowed_classes = {"14"}
    pairs = []
    current_dir = None

    for image_path in source_root.rglob("*.jpg"):
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()
            classes_in_file = {line.strip().split()[0] for line in lines}

        if not classes_in_file.issubset(allowed_classes):
            continue

        dir_of_image = image_path.parent
        if dir_of_image != current_dir:
            print(f"Verarbeite Ordner: {dir_of_image.relative_to(source_root)}")
            current_dir = dir_of_image

        pairs.append((image_path, label_path))

    return pairs


def copy_pair(image_path, label_path, split_dir):
    image_target = split_dir / "images" / image_path.name
    label_target = split_dir / "labels" / label_path.name
    shutil.copy2(image_path, image_target)
    shutil.copy2(label_path, label_target)

def split_dataset(pairs, target_dir, train_ratio):
    random.shuffle(pairs)
    split_index = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    for image, label in train_pairs:
        copy_pair(image, label, target_dir / "train")

    for image, label in val_pairs:
        copy_pair(image, label, target_dir / "val")

if __name__ == "__main__":
    print("Erzeuge Zielverzeichnisstruktur (falls nicht vorhanden)...")
    prepare_dirs(TARGET_DIR)

    print("Suche Bild-Label-Paare...")
    pairs = get_image_label_pairs(SOURCE_DIR)
    print(f"Gefundene Paare: {len(pairs)}")

    print("Teile in Training und Validierung...")
    split_dataset(pairs, TARGET_DIR, TRAIN_RATIO)

    print("Fertig.")
