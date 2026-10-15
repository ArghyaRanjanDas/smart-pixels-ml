#!/usr/bin/env bash
# Longer probe (100 epochs, otherwise the V-JEPA protocol) on a final checkpoint, run
# once that trainer has exited and freed its GPU memory. Separate output dir: the
# standard 20-epoch protocol result is kept as the headline number.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml; M=${1:?mask}; PID=${2:?trainer pid}
CK=$REPO/runs/o26_jepa/vjepa_$M/ckpt_ep0300.pt; O=$REPO/runs/o26_jepa/probe_${M}_ep0300_long100
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
until [ -f "$CK" ]; do sleep 60; done; sleep 30
mkdir -p "$O"
/work/pixi/global/.pixi/envs/default/bin/python $REPO/scripts/jepa/probe_attentive.py --ckpt "$CK" --out "$O" --epochs 100 > "$O/probe.log" 2>&1
echo "$(date +%F_%T) $M ep0300 LONG(100) $(grep BEST "$O/probe.log")" >> $REPO/runs/o26_jepa/probe_watcher.log
