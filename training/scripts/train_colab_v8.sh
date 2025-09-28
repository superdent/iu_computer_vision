# training/scripts/train_colab_v8.sh
#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/_common.sh"; [[ -f "$DATA_YAML" ]] || { echo "DATA_YAML not found: $DATA_YAML" >&2; exit 1; }

PLATFORM=colab; MODEL=v8; EPOCHS=1; BATCH=16; IMG=512; SEED=42
NAME="${MODEL}_${PLATFORM}_ep${EPOCHS}_b${BATCH}_img${IMG}_seed${SEED}"
PROJECT="$(cd "$(dirname "$0")/.." && pwd)/runs/${MODEL}/${PLATFORM}"

yolo detect train model=yolov8n.pt data="$DATA_YAML" epochs="$EPOCHS" imgsz="$IMG" batch="$BATCH" seed="$SEED" project="$PROJECT" name="$NAME"
