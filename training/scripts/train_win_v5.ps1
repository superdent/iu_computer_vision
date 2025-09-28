# training/scripts/train_win_v5.ps1
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\_common.ps1"; if (-not (Test-Path $DATA_YAML)) { Write-Error "DATA_YAML not found: $DATA_YAML"; exit 1 }

$PLATFORM="windows"; $MODEL="v5"; $EPOCHS=1; $BATCH=16; $IMG=512; $SEED=42
$NAME="$MODEL`_$PLATFORM`_ep$EPOCHS`_b$BATCH`_img$IMG`_seed$SEED"
Set-Location (Join-Path $PSScriptRoot "..\yolov5")
$PROJECT = (Resolve-Path (Join-Path $PSScriptRoot "..\runs\$MODEL\$PLATFORM"))

python train.py --weights yolov5n.pt --data $DATA_YAML --epochs $EPOCHS --imgsz $IMG --batch $BATCH --seed $SEED --project $PROJECT --name $NAME
