import os
import csv
import shutil
from PIL import Image
from pathlib import Path

# HART KODIERTE PFADANGABEN
INPUT_ROOT = Path(r"..\datasets\ppm")
OUTPUT_ROOT = Path(r"..\datasets\jpg")

def ppm_to_jpg(ppm_path, jpg_path):
    with Image.open(ppm_path) as img:
        rgb = img.convert('RGB')
        rgb.save(jpg_path)

def convert_annotation(csv_path, input_dir, output_dir):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            ppm_filename = row['Filename']
            label = int(row['ClassId'])

            input_image_path = input_dir / ppm_filename
            output_image_path = output_dir / ppm_filename.replace('.ppm', '.jpg')
            output_label_path = output_image_path.with_suffix('.txt')

            output_dir.mkdir(parents=True, exist_ok=True)
            ppm_to_jpg(input_image_path, output_image_path)

            with Image.open(output_image_path) as im:
                img_w, img_h = im.size

            x1 = int(row['Roi.X1'])
            y1 = int(row['Roi.Y1'])
            x2 = int(row['Roi.X2'])
            y2 = int(row['Roi.Y2'])

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            with open(output_label_path, 'w') as f:
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_dataset(input_root, output_root):
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    for dirpath, _, filenames in os.walk(input_root):
        for file in filenames:
            if file.startswith('GT-') and file.endswith('.csv'):
                csv_path = Path(dirpath) / file
                relative_path = csv_path.parent.relative_to(input_root)
                input_image_dir = csv_path.parent
                output_image_dir = output_root / relative_path
                print(f"Verarbeite Ordner: {relative_path}")
                convert_annotation(csv_path, input_image_dir, output_image_dir)

if __name__ == "__main__":
    if not INPUT_ROOT.exists():
        print("Quellpfad ungültig.")
        exit(1)
    print(f"Lösche Zielverzeichnis: {OUTPUT_ROOT}")
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)
    print("Fertig.")
