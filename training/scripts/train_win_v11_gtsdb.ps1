# training/scripts/train_win_v11.ps1
$ErrorActionPreference = "Stop"
$PLATFORM="windows"; $MODEL="v11"
$REPO_ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$PROJECT   = Join-Path $REPO_ROOT "runs\gtsdb\$MODEL\$PLATFORM"
$NAME      = "$MODEL`_$PLATFORM"
yolo detect train cfg=configs/yolov11_gtsdb.yaml project=$PROJECT name=$NAME
