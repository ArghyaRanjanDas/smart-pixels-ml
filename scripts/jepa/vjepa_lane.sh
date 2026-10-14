#!/usr/bin/env bash
# Run V-JEPA pretraining variants one after another (one GPU "lane").
# usage: vjepa_lane.sh <mask> [<mask> ...]   -- 300 epochs each (~88K iters, the paper's budget)
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/pixi/global/.pixi/envs/default/bin/python
for M in "$@"; do
  O=$REPO/runs/o26_jepa/vjepa_$M; mkdir -p "$O"
  echo "[lane] $(date +%F_%T) start $M"
  "$PY" "$REPO/scripts/jepa/pretrain_vjepa.py" --mask "$M" --epochs 300 --out "$O" >> "$O/train.log" 2>&1
  echo "[lane] $(date +%F_%T) end $M (exit $?)"
done
