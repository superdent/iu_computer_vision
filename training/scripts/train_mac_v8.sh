# training/scripts/train_mac_v8.sh
#!/usr/bin/env bash
set -euo pipefail

PLATFORM=mac; MODEL=v8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PROJECT="$REPO_ROOT/runs/${MODEL}/${PLATFORM}"
NAME="${MODEL}_${PLATFORM}"

CFG="$REPO_ROOT/configs/yolov8.yaml"
yolo detect train cfg="$CFG" project="$PROJECT" name="$NAME"
