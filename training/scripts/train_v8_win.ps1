# training/scripts/train_v8_win.ps1
$ErrorActionPreference = "Stop"
yolo train cfg=training/configs/yolov8.yaml
