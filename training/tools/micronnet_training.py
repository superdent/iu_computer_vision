# train_micronet.py
import os, csv, time
from pathlib import Path
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ===================== KONFIG =====================
DATA_ROOT = Path(r"C:\Develop\Python\PyCharmProjects\iu_computer_vision\training\datasets\gtsrb\micronnet_training")
TRAIN_CSV = DATA_ROOT / "train.csv"
VAL_CSV   = DATA_ROOT / "val.csv"
IMG_SIZE  = 48
N_CLASSES = 43
BATCH     = 256
EPOCHS    = 100
LR        = 1e-3
EARLY_STOP_PATIENCE = 10
OUT_DIR   = Path("./runs/runs_micronnet_windows")
AMP       = True  # Mixed Precision (aktiviert sich nur mit CUDA)
NUM_WORKERS = 4
# ==================================================

class CSVDataset(Dataset):
    def __init__(self, csv_file: Path, root: Path, transform=None):
        self.root = root
        self.transform = transform
        self.items: List[Tuple[Path,int]] = []
        with open(csv_file, newline="") as f:
            r = csv.reader(f)
            for row in r:
                fp, lab = row
                self.items.append((root / fp, int(lab)))
        if not self.items:
            raise RuntimeError(f"Keine Einträge in {csv_file}")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

class MicronNet(nn.Module):
    def __init__(self, n_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 48->24
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 24->12
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 12->6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("========== MicronNet Training ==========")
    print(f"[PATH] DATA_ROOT: {DATA_ROOT}")
    print(f"[CSV ] Train: {TRAIN_CSV}")
    print(f"[CSV ] Val  : {VAL_CSV}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] Device: {device} | CUDA: {use_cuda}")

    # Transforms
    tr = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    tr_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.RandomRotation(8, fill=(0,0,0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

    ds_train = CSVDataset(TRAIN_CSV, DATA_ROOT, transform=tr_train)
    ds_val   = CSVDataset(VAL_CSV,   DATA_ROOT, transform=tr)

    print(f"[DATA] Train: {len(ds_train)} | Val: {len(ds_val)}")

    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=use_cuda, persistent_workers=use_cuda)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=use_cuda, persistent_workers=use_cuda)

    model = MicronNet(N_CLASSES).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    # AMP nur nutzen, wenn CUDA vorhanden ist
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and use_cuda)

    def evaluate():
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        autocast_ctx = torch.amp.autocast("cuda" if use_cuda else "cpu", enabled=AMP and use_cuda)
        with torch.no_grad(), autocast_ctx:
            for x,y in dl_val:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = crit(logits, y)
                loss_sum += loss.item() * y.size(0)
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct/total, loss_sum/total

    print(f"[INFO] Params: {sum(p.numel() for p in model.parameters()):,}")
    best_acc, best_path = 0.0, None
    no_improve = 0
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        run_loss, seen = 0.0, 0
        autocast_ctx = torch.amp.autocast("cuda" if use_cuda else "cpu", enabled=AMP and use_cuda)

        for i,(x,y) in enumerate(dl_train, 1):
            x, y = x.to(device, non_blocking=use_cuda), y.to(device, non_blocking=use_cuda)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x)
                loss = crit(logits, y)
            if AMP and use_cuda:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            run_loss += loss.item() * y.size(0)
            seen     += y.size(0)
            if i % 20 == 0:
                print(f"[E{epoch:02d}] Step {i:04d}/{len(dl_train):04d} | loss {run_loss/seen:.4f}", flush=True)

        val_acc, val_loss = evaluate()
        sched.step(val_acc)
        dt = time.time() - t0
        print(f"[E{epoch:02d}] train_loss {run_loss/seen:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc*100:.2f}% | time {dt:.1f}s | lr {opt.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            best_path = OUT_DIR / f"micronnet_best_acc{best_acc*100:.2f}.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": best_acc,
                "config": {"IMG_SIZE": IMG_SIZE, "N_CLASSES": N_CLASSES,
                        "norm_mean": [0.5,0.5,0.5], "norm_std":[0.5,0.5,0.5]}
            }, best_path)
            print(f"[SAVE] Neuer Bestwert: {best_acc*100:.2f}%  => {best_path}")
        else:
            no_improve += 1
            print(f"[ES] Keine Verbesserung ({no_improve}/{EARLY_STOP_PATIENCE})")
            if no_improve >= EARLY_STOP_PATIENCE:
                print("[ES] Früher Abbruch – Plateau erreicht.")
                break

    print(f"[DONE] Best Val-Acc: {best_acc*100:.2f}% | Checkpoint: {best_path}")

if __name__ == "__main__":
    main()
