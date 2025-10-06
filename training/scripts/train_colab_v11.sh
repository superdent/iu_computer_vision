# training/scripts/train_colab_v11.sh
#!/usr/bin/env bash
set -euo pipefail
PLATFORM=colab; MODEL=v11
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PROJECT="$REPO_ROOT/runs/${MODEL}/${PLATFORM}"
NAME="${MODEL}_${PLATFORM}"
yolo detect train cfg=training/configs/yolov11.yaml project="$PROJECT" name="$NAME"
