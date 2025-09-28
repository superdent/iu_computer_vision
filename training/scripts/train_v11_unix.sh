# training/scripts/train_v11_unix.sh
#!/usr/bin/env bash
set -euo pipefail
yolo train cfg=training/configs/yolov11.yaml
