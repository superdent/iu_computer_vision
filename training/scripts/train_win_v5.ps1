# training/scripts/train_win_v5.ps1
$ErrorActionPreference = "Stop"

$PLATFORM="windows"; $MODEL="v5"; $EPOCHS=1; $BATCH=16; $IMG=512; $SEED=42
$NAME="$MODEL`_$PLATFORM`_ep$EPOCHS`_b$BATCH`_img$IMG`_seed$SEED"

# Pfade CWD-sicher auflösen (ohne _common.ps1)
$REPO_ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$DATA_V5_YAML = Resolve-Path (Join-Path $REPO_ROOT "configs\datav5.yaml")
if (-not (Test-Path $DATA_V5_YAML)) { Write-Error "ERROR: DATA_V5_YAML not found: $DATA_V5_YAML"; exit 1 }

Set-Location (Join-Path $REPO_ROOT "yolov5")
$PROJECT = Resolve-Path (Join-Path $REPO_ROOT "runs\$MODEL\$PLATFORM")

python train.py `
  --weights yolov5n.pt `
  --data $DATA_V5_YAML `
  --epochs $EPOCHS --imgsz $IMG --batch $BATCH --seed $SEED `
  --project $PROJECT --name $NAME
