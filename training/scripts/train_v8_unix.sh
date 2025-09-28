# training/scripts/train_v8_unix.sh
#!/usr/bin/env bash
set -euo pipefail
yolo train cfg=training/configs/yolov8.yaml
