# training/scripts/train_mac_v11.sh
#!/usr/bin/env bash
set -euo pipefail

PLATFORM=mac; MODEL=v11
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PROJECT="$REPO_ROOT/runs/${MODEL}/${PLATFORM}"
NAME="${MODEL}_${PLATFORM}"

CFG="$REPO_ROOT/configs/yolov11.yaml"
yolo detect train cfg="$CFG" project="$PROJECT" name="$NAME"
