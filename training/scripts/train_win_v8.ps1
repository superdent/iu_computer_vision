# training/scripts/train_win_v8.ps1
$ErrorActionPreference = "Stop"
$PLATFORM="windows"; $MODEL="v8"
$REPO_ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$PROJECT   = Join-Path $REPO_ROOT "runs\$MODEL\$PLATFORM"
$NAME      = "$MODEL`_$PLATFORM"
yolo detect train cfg=configs/yolov8.yaml project=$PROJECT name=$NAME
