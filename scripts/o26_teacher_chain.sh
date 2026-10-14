#!/usr/bin/env bash
# O26 step 1a: the FULL-VIEW teacher. ViT_MaxDeep (deep head, NO router), float
# 101-slice input, O25 loss stack (binned NLL by-term + 60 zero-bias MDMM
# constraints). best.weights picked on val_plain_nll -- O23's full-view arms peaked
# at ep 300-900 and then overfit, so the plain-NLL optimum must be the one kept.
# Same seed as O23 arm A (30042): the only difference from that run is the loss.
#   usage: SEED TOTAL CHUNK [GPUMB]
set -u
SEED=${1:-30042}; TOTAL=${2:-2000}; CHUNK=${3:-100}; GPUMB=${4:-12000}
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
OUT=${OUTROOT:-$REPO/runs/o26_jepa/teacher_full101}/s$SEED
mkdir -p "$OUT"
cur_ep () { local h; h=$(ls "$OUT"/seed_*/history.csv 2>/dev/null | head -1)
            [ -n "$h" ] && echo $(( $(wc -l < "$h") - 1 )) || echo 0; }
cur=$(cur_ep)
echo "[teacher $SEED] start ep $cur -> $TOTAL, chunk $CHUNK"
while [ "$cur" -lt "$TOTAL" ]; do
  next=$(( cur + CHUNK )); [ "$next" -gt "$TOTAL" ] && next=$TOTAL
  env SMARTPIX_MODEL_NAME=ViT_MaxDeep SMARTPIX_FLOAT_INPUT=1 SMARTPIX_LOSS_V2=1 \
      SMARTPIX_ZEROBIAS=1 SMARTPIX_BINBAL=byterm SMARTPIX_NBINS=15 \
      SMARTPIX_BIAS_TOL=0.17 SMARTPIX_BIAS_MAXLAM=5.0 SMARTPIX_BIAS_INFCAP=2.0 \
      SMARTPIX_CKPT_MONITOR=val_plain_nll SMARTPIX_ABORT_THR=1e12 SMARTPIX_GPU_MEM_MB="$GPUMB" \
      "$PY" "$REPO/scripts/distillation/run_simplerouter_mdmm_discovery.py" \
        --epochs "$next" --target 1 --seeds "$SEED" --out "$OUT" --extend \
        >> "$OUT/chain.log" 2>&1
  new=$(cur_ep)
  [ "$new" -le "$cur" ] && { echo "[teacher $SEED] NO PROGRESS at $cur (wanted $next) -- stopping"; exit 1; }
  cur=$new
done
echo "[teacher $SEED] DONE at $cur"
