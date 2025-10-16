# visualize_yolo_labels.py
# Visualizes the first N YOLO images and their bounding boxes
# Compatible with Windows + Ultralytics datasets

import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# === CONFIGURATION ===
DATA_ROOT = Path(r"datasets\gtsdb\yolo_training")
SPLIT = "val"           # "train" or "val"
COUNT = 10              # number of images to visualize
OUT_DIR = Path(r"tools\label_viz")  # output folder for visualizations
# ======================

def load_labels(label_file: Path):
    boxes = []
    if not label_file.exists():
        return boxes
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                boxes.append((cls, x, y, w, h))
            except ValueError:
                continue
    return boxes

def draw_boxes(img_path: Path, labels, out_path: Path):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    if not labels:
        draw.text((10, 10), "NO LABELS", fill=(255, 0, 0), font=font)
    else:
        for (cls, xc, yc, w, h) in labels:
            x1 = int((xc - w / 2) * W)
            y1 = int((yc - h / 2) * H)
            x2 = int((xc + w / 2) * W)
            y2 = int((yc + h / 2) * H)
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
            label = f"class {cls}"
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    print(f"Saved: {out_path}")

def main():
    img_dir = DATA_ROOT / SPLIT / "images"
    lbl_dir = DATA_ROOT / SPLIT / "labels"

    images = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not images:
        print(f"No images found in {img_dir}")
        return

    sample = images[:COUNT]
    print(f"Visualizing {len(sample)} images from {img_dir}...")

    for img_path in sample:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        labels = load_labels(label_path)
        out_file = OUT_DIR / f"{SPLIT}_{img_path.name}"
        draw_boxes(img_path, labels, out_file)

if __name__ == "__main__":
    main()
