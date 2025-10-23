# training/scripts/train_win_v11_fussgaenger.ps1
$ErrorActionPreference = "Stop"
$PLATFORM="windows"; $MODEL="v11"
$REPO_ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$PROJECT   = Join-Path $REPO_ROOT "runs\gtsdb\$MODEL\$PLATFORM"
$NAME      = "$MODEL`_$PLATFORM"
yolo detect train cfg=configs/yolov11_fussgaenger.yaml project=$PROJECT name=$NAME
