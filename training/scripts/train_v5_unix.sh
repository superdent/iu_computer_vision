# training/scripts/train_v5_unix.sh
#!/usr/bin/env bash
set -euo pipefail
python -m yolov5 train --weights yolov5n.pt --data training/configs/data.yaml --epochs 5 --imgsz 640 --batch 16 --seed 42 --project training/runs/v5 --name exp
