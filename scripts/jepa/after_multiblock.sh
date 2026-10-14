#!/usr/bin/env bash
# Start randtube as soon as the multiblock trainer exits and frees its 22.5 GB.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml; PID=${1:?multiblock trainer pid}
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
O=$REPO/runs/o26_jepa/vjepa_randtube; mkdir -p "$O"
echo "$(date +%F_%T) multiblock exited; starting randtube accum=2" >> $REPO/runs/o26_jepa/lane.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /work/pixi/global/.pixi/envs/default/bin/python $REPO/scripts/jepa/pretrain_vjepa.py \
  --mask randtube --epochs 300 --batch 512 --accum 2 --out "$O" >> "$O/train.log" 2>&1
