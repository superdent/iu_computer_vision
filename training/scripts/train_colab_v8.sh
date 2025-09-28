# training/scripts/train_colab_v8.sh
#!/usr/bin/env bash
set -euo pipefail
PLATFORM=colab; MODEL=v8; EPOCHS=5; BATCH=16; IMG=640; SEED=42
NAME="${MODEL}_${PLATFORM}_ep${EPOCHS}_b${BATCH}_img${IMG}_seed${SEED}"
PROJECT="training/runs/${MODEL}/${PLATFORM}"
yolo detect train model=yolov8n.pt data=training/configs/data.yaml epochs=$EPOCHS imgsz=$IMG batch=$BATCH seed=$SEED project="$PROJECT" name="$NAME"
