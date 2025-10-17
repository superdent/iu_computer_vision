# training/scripts/train_colab_v11.sh
#!/usr/bin/env bash
set -euo pipefail
PLATFORM=colab; MODEL=v11
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PROJECT="$REPO_ROOT/runs/${MODEL}/${PLATFORM}"
NAME="${MODEL}_${PLATFORM}"
pwd
yolo classify train cfg=$REPO_ROOT/configs/yolov11_colab.yaml project="$PROJECT" name="$NAME"
