import os
import csv
import shutil
from PIL import Image
from pathlib import Path
from collections import defaultdict

# HART KODIERTE PFADANGABEN
INPUT_ROOT = Path("./datasets/ppm")
OUTPUT_ROOT = Path("./datasets/yolo_training")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
CLASS_FILTER = []  # z.B. [0, 1, 14] oder [] für alle Klassen

def ppm_to_jpg(ppm_path, jpg_path):
    with Image.open(ppm_path) as img:
        rgb = img.convert('RGB')
        rgb.save(jpg_path)

def read_annotations(input_root):
    annotations = []
    for dirpath, _, filenames in os.walk(input_root):
        for file in filenames:
            if file.startswith('GT-') and file.endswith('.csv'):
                csv_path = Path(dirpath) / file
                with open(csv_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile, delimiter=';')
                    for row in reader:
                        class_id = int(row['ClassId'])
                        if CLASS_FILTER and class_id not in CLASS_FILTER:
                            continue
                        row['ClassId'] = class_id
                        row['FullPath'] = Path(dirpath) / row['Filename']
                        annotations.append(row)
    return annotations

def split_annotations(annotations):
    from random import Random
    rand = Random(RANDOM_SEED)

    class_to_annots = defaultdict(list)
    for row in annotations:
        class_to_annots[row['ClassId']].append(row)

    train_set, val_set = [], []
    for class_id, items in class_to_annots.items():
        rand.shuffle(items)
        split_point = int(len(items) * TRAIN_RATIO)
        train_set.extend(items[:split_point])
        val_set.extend(items[split_point:])

    return train_set, val_set

def prepare_output_dirs(root):
    for split in ['train', 'val']:
        for sub in ['images', 'labels']:
            path = root / split / sub
            path.mkdir(parents=True, exist_ok=True)

def convert_and_write(rows, output_root, split):
    for row in rows:
        ppm_path = row['FullPath']
        label = row['ClassId']
        jpg_name = ppm_path.name.replace('.ppm', '.jpg')

        output_img_path = output_root / split / 'images' / jpg_name
        output_lbl_path = output_root / split / 'labels' / jpg_name.replace('.jpg', '.txt')

        ppm_to_jpg(ppm_path, output_img_path)

        with Image.open(output_img_path) as im:
            img_w, img_h = im.size

        x1, y1 = int(row['Roi.X1']), int(row['Roi.Y1'])
        x2, y2 = int(row['Roi.X2']), int(row['Roi.Y2'])

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        with open(output_lbl_path, 'w') as f:
            f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_dataset(input_root, output_root):
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    print("Lese Annotationen...")
    annotations = read_annotations(input_root)
    print(f"Gefundene Annotationen: {len(annotations)}")

    print("Splitte Daten klassenweise und deterministisch...")
    train_set, val_set = split_annotations(annotations)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    print("Bereite Zielverzeichnisse vor...")
    prepare_output_dirs(output_root)

    print("Verarbeite Trainingsdaten...")
    convert_and_write(train_set, output_root, 'train')

    print("Verarbeite Validierungsdaten...")
    convert_and_write(val_set, output_root, 'val')

if __name__ == "__main__":
    if not INPUT_ROOT.exists():
        print("Quellpfad ungültig.")
        exit(1)
    print(f"Lösche Zielverzeichnis: {OUTPUT_ROOT}")
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)
    print("Fertig.")
