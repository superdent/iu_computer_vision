# training/scripts/train_v11_win.ps1
$ErrorActionPreference = "Stop"
yolo train cfg=training/configs/yolov11.yaml
