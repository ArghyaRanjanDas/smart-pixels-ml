#!/usr/bin/env bash
# Probe V-JEPA checkpoints in the order they APPEAR (variants now train in parallel):
# standard 20-epoch probe + collapse diagnostic at epochs 50/100/200/300, and the
# 100-epoch long probe on each variant's final epoch-300 encoder.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml; R=$REPO/runs/o26_jepa
PY=/work/pixi/global/.pixi/envs/default/bin/python
todo () {   # print "mtime kind mask epoch" for every existing checkpoint still unprobed
  for M in multiblock temporal causal randtube; do for E in 0050 0100 0200 0300; do
    CK=$R/vjepa_$M/ckpt_ep$E.pt; [ -f "$CK" ] || continue
    [ -f "$R/probe_${M}_ep$E/probe.json" ] || echo "$(stat -c %Y "$CK") std $M $E"
    [ "$E" = 0300 ] && { [ -f "$R/probe_${M}_ep0300_long100/probe.json" ] || echo "$(stat -c %Y "$CK") long $M $E"; }
  done; done | sort -n
}
total () { echo 20; }      # 16 standard + 4 long
while :; do
  N=$(ls $R/probe_*_ep*/probe.json 2>/dev/null | wc -l); [ "$N" -ge "$(total)" ] && break
  next=$(todo | head -1)
  if [ -z "$next" ]; then sleep 120; continue; fi
  set -- $next; KIND=$2; M=$3; E=$4; CK=$R/vjepa_$M/ckpt_ep$E.pt; sleep 30
  if [ "$KIND" = std ]; then
    O=$R/probe_${M}_ep$E; mkdir -p "$O"
    $PY $REPO/scripts/jepa/diag_collapse.py "$CK" > "$O/collapse.txt" 2>&1
    $PY $REPO/scripts/jepa/probe_attentive.py --ckpt "$CK" --out "$O" > "$O/probe.log" 2>&1
    echo "$(date +%F_%T) $M ep$E $(grep BEST "$O/probe.log") | $(grep 'target enc' "$O/collapse.txt")" >> $R/probe_watcher.log
  else
    O=$R/probe_${M}_ep0300_long100; mkdir -p "$O"
    $PY $REPO/scripts/jepa/probe_attentive.py --ckpt "$CK" --out "$O" --epochs 100 > "$O/probe.log" 2>&1
    echo "$(date +%F_%T) $M ep0300 LONG(100) $(grep BEST "$O/probe.log")" >> $R/probe_watcher.log
  fi
done
