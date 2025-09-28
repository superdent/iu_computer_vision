# training/scripts/train_mac_v5.sh
#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/_common.sh"; [[ -f "$DATA_YAML" ]] || { echo "DATA_YAML not found: $DATA_YAML" >&2; exit 1; }

PLATFORM=mac; MODEL=v5; EPOCHS=1; BATCH=16; IMG=512; SEED=42
NAME="${MODEL}_${PLATFORM}_ep${EPOCHS}_b${BATCH}_img${IMG}_seed${SEED}"
cd "$(dirname "$0")/../yolov5"
PROJECT="$(cd .. && pwd)/runs/${MODEL}/${PLATFORM}"

python train.py --weights yolov5n.pt --data "$DATA_YAML" --epochs "$EPOCHS" --imgsz "$IMG" --batch "$BATCH" --seed "$SEED" --project "$PROJECT" --name "$NAME"
