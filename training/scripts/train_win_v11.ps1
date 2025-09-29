# training/scripts/train_win_v11.ps1
$ErrorActionPreference = "Stop"
$PLATFORM="windows"; $MODEL="v11"
$REPO_ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$PROJECT   = Resolve-Path (Join-Path $REPO_ROOT "runs\$MODEL\$PLATFORM")
$NAME      = "$MODEL`_$PLATFORM"
yolo detect train cfg=training/configs/yolov11.yaml project=$PROJECT name=$NAME
