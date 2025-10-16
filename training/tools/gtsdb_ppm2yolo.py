import os
import csv
import shutil
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter

# -------------------- KONFIG --------------------
INPUT_ROOT = Path("./datasets/gtsdb/FullIJCNN2013")
GT_PATH = INPUT_ROOT / "gt.txt"

OUTPUT_ROOT = Path("./datasets/gtsdb/yolo_training")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
CLASS_FILTER = []  # z.B. [0, 1, 14] oder [] für alle Klassen
CONVERT_TO_JPG = True
ENFORCE_IMAGE_EXCLUSIVITY = True  # verhindert, dass dasselbe Bild in Train UND Val landet
# ------------------------------------------------

def ppm_to_jpg(ppm_path, jpg_path):
    with Image.open(ppm_path) as img:
        rgb = img.convert('RGB')
        rgb.save(jpg_path, quality=95)

def read_gt(gt_path, images_dir):
    """
    Erwartetes gt.txt-Format (Semikolon-getrennt):
    Filename;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
    Mehrere Zeilen pro Bild möglich.
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt nicht gefunden unter: {gt_path}")

    annotations = []
    with open(gt_path, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) != 6:
                continue
            filename, x1, y1, x2, y2, class_id = row
            class_id = int(class_id)
            if CLASS_FILTER and class_id not in CLASS_FILTER:
                continue
            ppm_path = images_dir / filename
            annotations.append({
                "Filename": filename,
                "FullPath": ppm_path,
                "Roi.X1": int(x1),
                "Roi.Y1": int(y1),
                "Roi.X2": int(x2),
                "Roi.Y2": int(y2),
                "ClassId": class_id
            })
    return annotations

def split_classwise_deterministic(annotations, train_ratio=0.8, seed=42):
    """
    1) Klassenweise deterministisch splitten (Zeilenebene).
    2) Optional: Bild-Exklusivität erzwingen (kein Leakage).
       -> Konfliktlösung: Bild geht in den Split, in dem es mehr Zeilen hat (Tie-Break via Seed).
    """
    from random import Random
    rand = Random(seed)

    # Schritt 1: klassenweise Split auf Zeilenebene
    class_to_rows = defaultdict(list)
    for r in annotations:
        class_to_rows[r["ClassId"]].append(r)

    initial_train, initial_val = [], []
    for cls, rows in class_to_rows.items():
        rows_sorted = list(rows)
        rand.shuffle(rows_sorted)
        k = int(len(rows_sorted) * train_ratio)
        initial_train.extend(rows_sorted[:k])
        initial_val.extend(rows_sorted[k:])

    if not ENFORCE_IMAGE_EXCLUSIVITY:
        return initial_train, initial_val

    # Schritt 2: Bild-Exklusivität erzwingen
    def index_by_image(rows):
        by_img = defaultdict(list)
        for r in rows:
            by_img[r["Filename"]].append(r)
        return by_img

    train_by_img = index_by_image(initial_train)
    val_by_img = index_by_image(initial_val)

    # Finde Konflikt-Bilder (in beiden Splits)
    conflict_imgs = set(train_by_img.keys()) & set(val_by_img.keys())

    # Ergebniscontainer
    final_train, final_val = [], []

    # Start mit allen konfliktfreien Bildern
    for fname, rows in train_by_img.items():
        if fname not in conflict_imgs:
            final_train.extend(rows)
    for fname, rows in val_by_img.items():
        if fname not in conflict_imgs:
            final_val.extend(rows)

    # Konflikte deterministisch lösen
    for fname in sorted(conflict_imgs):
        t_rows = train_by_img[fname]
        v_rows = val_by_img[fname]
        if len(t_rows) > len(v_rows):
            winner = "train"
        elif len(v_rows) > len(t_rows):
            winner = "val"
        else:
            # Tie-Break deterministisch per Seed
            winner = "train" if rand.random() < train_ratio else "val"

        if winner == "train":
            final_train.extend(t_rows + v_rows)  # alle Zeilen dieses Bildes nach train
        else:
            final_val.extend(t_rows + v_rows)    # alle Zeilen dieses Bildes nach val

    return final_train, final_val

def prepare_output_dirs(root):
    for split in ['train', 'val']:
        for sub in ['images', 'labels']:
            path = root / split / sub
            path.mkdir(parents=True, exist_ok=True)

def group_rows_by_image(rows):
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["Filename"]].append(r)
    return by_img

def write_split(split_name, rows, output_root):
    by_img = group_rows_by_image(rows)
    count_img = 0
    collisions = 0

    for filename, rlist in by_img.items():
        ppm_path = rlist[0]["FullPath"]
        prefix = Path(ppm_path).parent.name  # Ordnername (z.B. 'ppm')
        stem = Path(filename).stem

        jpg_name = f"{prefix}_{stem}.jpg" if CONVERT_TO_JPG else f"{prefix}_{stem}.ppm"
        txt_name = f"{prefix}_{stem}.txt"

        out_img_path = output_root / split_name / "images" / jpg_name
        out_lbl_path = output_root / split_name / "labels" / txt_name

        if out_img_path.exists() or out_lbl_path.exists():
            print(f"Warnung: {jpg_name} wird überschrieben!")
            collisions += 1

        # Bild konvertieren/kopieren
        if CONVERT_TO_JPG:
            ppm_to_jpg(ppm_path, out_img_path)
            with Image.open(out_img_path) as im:
                img_w, img_h = im.size
        else:
            # PPM unverändert kopieren
            shutil.copy2(ppm_path, out_img_path)
            with Image.open(ppm_path) as im:
                img_w, img_h = im.size

        # Labels schreiben (alle Boxen des Bildes)
        with open(out_lbl_path, "w") as f:
            for row in rlist:
                label = row["ClassId"]
                x1, y1 = row["Roi.X1"], row["Roi.Y1"]
                x2, y2 = row["Roi.X2"], row["Roi.Y2"]
                # YOLO-Normalisierung
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        count_img += 1

    print(f"{split_name.capitalize()}: {count_img} Bilder geschrieben, {collisions} Kollisionen")

def print_class_stats(label_rows, title):
    cnt = Counter(r["ClassId"] for r in label_rows)
    total = sum(cnt.values())
    print(f"\n[{title}] Zeilen pro Klasse und Anteil:")
    for cls in sorted(cnt):
        n = cnt[cls]
        ratio = n / total if total else 0
        print(f"  Klasse {cls:>2}: {n:>4}  ({ratio:6.2%})")
    print(f"  Gesamt: {total}\n")

def process_dataset():
    if not INPUT_ROOT.exists():
        print(f"Quellpfad ungültig: {INPUT_ROOT}")
        return
    if not GT_PATH.exists():
        print(f"gt.txt nicht gefunden: {GT_PATH}")
        return

    if OUTPUT_ROOT.exists():
        print(f"Lösche Zielverzeichnis: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Lese gt.txt ...")
    annotations = read_gt(GT_PATH, INPUT_ROOT)
    print(f"Gefundene Annotationen (Zeilen): {len(annotations)}")
    print(f"Einzigartige Bilder: {len(set(a['Filename'] for a in annotations))}")

    print("Splitte klassenweise, deterministisch, und erzwinge Bild-Exklusivität ...")
    train_rows, val_rows = split_classwise_deterministic(
        annotations, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED
    )
    print(f"Train-Zeilen: {len(train_rows)}, Val-Zeilen: {len(val_rows)}")

    print_class_stats(train_rows, "TRAIN")
    print_class_stats(val_rows, "VAL")

    print("Bereite Zielverzeichnisse vor ...")
    prepare_output_dirs(OUTPUT_ROOT)

    print("Schreibe Trainingssplit ...")
    write_split("train", train_rows, OUTPUT_ROOT)

    print("Schreibe Validierungssplit ...")
    write_split("val", val_rows, OUTPUT_ROOT)

    print("Fertig.")

if __name__ == "__main__":
    process_dataset()
