#!/usr/bin/env bash
# Probe each V-JEPA variant's checkpoints as they appear: frozen-encoder attentive
# probe (probe_attentive.py) + collapse diagnostic (diag_collapse.py).
# Checkpoints at epochs 50/100/200/300; variants in lane order.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/pixi/global/.pixi/envs/default/bin/python
for M in multiblock temporal causal randtube; do
  for E in 0050 0100 0200 0300; do
    CK=$REPO/runs/o26_jepa/vjepa_$M/ckpt_ep$E.pt; O=$REPO/runs/o26_jepa/probe_${M}_ep$E
    [ -f "$O/probe.json" ] && continue
    until [ -f "$CK" ]; do sleep 120; done; sleep 30          # let the save finish
    mkdir -p "$O"
    "$PY" "$REPO/scripts/jepa/diag_collapse.py" "$CK" > "$O/collapse.txt" 2>&1
    "$PY" "$REPO/scripts/jepa/probe_attentive.py" --ckpt "$CK" --out "$O" > "$O/probe.log" 2>&1
    echo "$(date +%F_%T) $M ep$E $(grep BEST "$O/probe.log") | $(grep 'target enc' "$O/collapse.txt")"
  done
done
