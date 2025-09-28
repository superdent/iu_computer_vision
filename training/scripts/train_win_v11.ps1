# training/scripts/train_win_v11.ps1
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\_common.ps1"; if (-not (Test-Path $DATA_YAML)) { Write-Error "DATA_YAML not found: $DATA_YAML"; exit 1 }

$PLATFORM="windows"; $MODEL="v11"; $EPOCHS=1; $BATCH=16; $IMG=512; $SEED=42
$NAME="$MODEL`_$PLATFORM`_ep$EPOCHS`_b$BATCH`_img$IMG`_seed$SEED"
$PROJECT = (Resolve-Path (Join-Path $PSScriptRoot "..\runs\$MODEL\$PLATFORM"))

yolo detect train model=yolo11n.pt data=$DATA_YAML epochs=$EPOCHS imgsz=$IMG batch=$BATCH seed=$SEED project=$PROJECT name=$NAME
