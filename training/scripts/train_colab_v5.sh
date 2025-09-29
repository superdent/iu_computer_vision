# training/scripts/train_colab_v5.sh
#!/usr/bin/env bash
set -euo pipefail

PLATFORM=colab; MODEL=v5; EPOCHS=1; BATCH=16; IMG=512; SEED=42
NAME="${MODEL}_${PLATFORM}_ep${EPOCHS}_b${BATCH}_img${IMG}_seed${SEED}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_V5_YAML="$REPO_ROOT/configs/datav5.yaml"
[[ -f "$DATA_V5_YAML" ]] || { echo "ERROR: DATA_V5_YAML not found: $DATA_V5_YAML" >&2; exit 1; }

cd "$REPO_ROOT/yolov5"
PROJECT="$REPO_ROOT/runs/${MODEL}/${PLATFORM}"

python train.py \
  --weights yolov5n.pt \
  --data "$DATA_V5_YAML" \
  --epochs "$EPOCHS" --imgsz "$IMG" --batch "$BATCH" --seed "$SEED" \
  --project "$PROJECT" --name "$NAME"
