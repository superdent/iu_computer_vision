# training/scripts/train_win_v5.ps1
$ErrorActionPreference = "Stop"
$PLATFORM="windows"; $MODEL="v5"; $EPOCHS=5; $BATCH=16; $IMG=640; $SEED=42
$NAME="$MODEL`_$PLATFORM`_ep$EPOCHS`_b$BATCH`_img$IMG`_seed$SEED"
$PROJECT="training/runs/$MODEL/$PLATFORM"
python training/yolov5/train.py --weights yolov5n.pt --data training/configs/data.yaml --epochs $EPOCHS --imgsz $IMG --batch $BATCH --seed $SEED --project $PROJECT --name $NAME
