import csv, math, sys, time
from pathlib import Path
from PIL import Image

# ===================== KONFIG =====================
root = Path(r"C:\Develop\Python\PyCharmProjects\iu_computer_vision\training\datasets\gtsrb")
src  = root / "ppm"                          # erwartet: ppm/00000..00042/*.ppm
out  = root / "micronnet_training"           # Ziel: micronnet_training/{train,val}/{classId}/
img_size = 48
val_ratio = 0.2
# =================================================

def log(msg): 
    print(msg, flush=True)

def letterbox(im: Image.Image, size=48):
    im = im.convert("RGB")
    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    im_res = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im_res, ((size - nw)//2, (size - nh)//2))
    return canvas

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def unique_name(p: Path, cls: int):
    return f"{cls:05d}_{p.stem}.png"

def collect_by_class(src_dir: Path):
    if not src_dir.is_dir():
        log(f"[FATAL] Quelle nicht gefunden: {src_dir}")
        sys.exit(2)
    by_cls = {}
    total = 0
    for cls_dir in sorted(src_dir.iterdir()):
        if not cls_dir.is_dir(): 
            continue
        try:
            cls = int(cls_dir.name)
        except ValueError:
            log(f"[WARN] Überspringe Ordner (kein Klassen-Int): {cls_dir.name}")
            continue
        paths = sorted([p for p in cls_dir.glob("*.ppm")])
        if not paths:
            log(f"[WARN] Keine PPM-Dateien in: {cls_dir}")
            continue
        by_cls[cls] = paths
        total += len(paths)
        log(f"[INFO] Klasse {cls:05d}: {len(paths)} Bilder gefunden")
    if not by_cls:
        log("[FATAL] Keine Klassen/Bilder gefunden.")
        sys.exit(3)
    log(f"[INFO] Gesamt gefunden: {total} Bilder in {len(by_cls)} Klassen")
    return by_cls, total

def split_by_class(by_cls: dict, ratio: float):
    train, val = [], []
    log(f"[INFO] Starte deterministischen Split (Val-Ratio={ratio:.2f})")
    for c, paths in sorted(by_cls.items()):
        n = len(paths)
        n_val = int(round(n * ratio))
        v = paths[:n_val]
        t = paths[n_val:]
        val.extend((p, c) for p in v)
        train.extend((p, c) for p in t)
        log(f"[SPLIT] Klasse {c:05d}: Train={len(t):5d}  Val={len(v):5d}  (Total={n})")
    log(f"[INFO] Split fertig: Train={len(train)}  Val={len(val)}")
    return train, val

def process(split: str, items, out_root: Path):
    split_dir = out_root / split
    ensure_dir(split_dir)
    rows = []
    ok, err = 0, 0
    start = time.time()
    per_class_count = {}

    log(f"[PROC] Erzeuge {split.upper()}-Bilder unter: {split_dir}")
    for idx, (p, cls) in enumerate(items, 1):
        try:
            cls_dir = split_dir / f"{cls:05d}"
            ensure_dir(cls_dir)
            img = Image.open(p)
            img = letterbox(img, img_size)
            fname = unique_name(p, cls)
            dst = cls_dir / fname
            img.save(dst, format="PNG")
            rows.append([str(dst.relative_to(out)).replace("\\","/"), cls])
            ok += 1
            per_class_count[cls] = per_class_count.get(cls, 0) + 1
            if ok % 1000 == 0:
                log(f"[{split.upper()}] {ok:6d} gespeichert…")
        except Exception as e:
            err += 1
            log(f"[ERROR] {split} -> {p}  ({e})")

    csv_path = out_root / f"{split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)

    dur = time.time() - start
    log(f"[DONE] {split.upper()}: OK={ok}  ERR={err}  Dauer={dur:.1f}s  CSV={csv_path}")
    log(f"[STATS] {split.upper()} pro Klasse: " +
        ", ".join([f"{k:05d}:{v}" for k,v in sorted(per_class_count.items())]))
    return ok, err, csv_path

def main():
    log("========== GTSRB → MicronNet-Preproc (48×48, deterministisch) ==========")
    log(f"[PATH] Quelle: {src}")
    log(f"[PATH] Ziel:   {out}")
    ensure_dir(out)

    by_cls, total = collect_by_class(src)
    train_set, val_set = split_by_class(by_cls, val_ratio)

    n_train_ok, n_train_err, train_csv = process("train", train_set, out)
    n_val_ok,   n_val_err,   val_csv   = process("val",   val_set,   out)

    log("=========== ZUSAMMENFASSUNG ===========")
    log(f"Train  : OK={n_train_ok}  ERR={n_train_err}  CSV={train_csv}")
    log(f"Val    : OK={n_val_ok}    ERR={n_val_err}    CSV={val_csv}")
    log(f"Output : {out}")
    log("=======================================")

if __name__ == "__main__":
    main()
