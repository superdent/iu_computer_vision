# training/scripts/train_mac_v11.sh
#!/usr/bin/env bash
set -euo pipefail
PLATFORM=mac; MODEL=v11; EPOCHS=5; BATCH=16; IMG=640; SEED=42
NAME="${MODEL}_${PLATFORM}_ep${EPOCHS}_b${BATCH}_img${IMG}_seed${SEED}"
PROJECT="training/runs/${MODEL}/${PLATFORM}"
yolo detect train model=yolo11n.pt data=training/configs/data.yaml epochs=$EPOCHS imgsz=$IMG batch=$BATCH seed=$SEED project="$PROJECT" name="$NAME"
