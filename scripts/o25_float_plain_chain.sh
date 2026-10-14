#!/usr/bin/env bash
# O25 chunk chain for one seed.  usage: o25_chunk_run.sh SEED TOTAL CHUNK
#
# Why a chain at all: the driver leaks ~56-93 MB per epoch (measured on seed
# 50142: 4.05 GB @ 12 ep -> 7.83 GB @ 80 ep).  The ViT path leaks ~345 MB/ep
# through the same driver, so this is the driver, not QConv2D.  10,000 epochs in
# one process is ~600-900 GB and cannot run.  A chunk bounds the leak: RSS is
# released at process exit and the next chunk resumes from last.weights.hdf5.
#
# This is faithful, not a compromise: --epochs is an ABSOLUTE target, resume
# restores weights + optimizer + MDMM lambdas, logs append, and every schedule
# is keyed off the absolute epoch.  Same machinery the Gautschi chains used.
#
# SMARTPIX_PRESLICE_CACHE is what makes it affordable -- without it each restart
# re-reads the 101-slice TFRecords (~2-3 min, ~20 GB peak) to keep 0.39 GB.
set -u
SEED=${1:?seed}; TOTAL=${2:-10000}; CHUNK=${3:-150}
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
OUT=$REPO/runs/o25_float_plain/s$SEED
CACHE=$REPO/runs/o25_qconv2d/_preslice_cache
mkdir -p "$OUT" "$CACHE"

cur_ep () { local h; h=$(ls "$OUT"/seed_*/history.csv 2>/dev/null | head -1)
            [ -n "$h" ] && echo $(( $(wc -l < "$h") - 1 )) || echo 0; }

cur=$(cur_ep)
echo "[chain $SEED] start at epoch $cur, target $TOTAL, chunk $CHUNK"
while [ "$cur" -lt "$TOTAL" ]; do
  next=$(( cur + CHUNK )); [ "$next" -gt "$TOTAL" ] && next=$TOTAL
  echo "[chain $SEED] $(date +%H:%M:%S) chunk -> $next"
  env SMARTPIX_MODEL_NAME=Conv2D_Max SMARTPIX_LOSS_V2=1 SMARTPIX_PRESLICE=10,21 \
      SMARTPIX_PRESLICE_CACHE="$CACHE" \
      \
      \
      SMARTPIX_INIT_THR=8.565593719482422,20.227340698242188,47.49168395996094 \
      SMARTPIX_FREEZE_THR=1 SMARTPIX_ABORT_THR=1e12 SMARTPIX_GPU_MEM_MB=3500 \
      "$PY" "$REPO/scripts/distillation/run_simplerouter_mdmm_discovery.py" \
        --epochs "$next" --target 1 --seeds "$SEED" --out "$OUT" --extend \
        >> "$OUT/chain.log" 2>&1
  new=$(cur_ep)
  if [ "$new" -le "$cur" ]; then
    echo "[chain $SEED] NO PROGRESS at epoch $cur (wanted $next) -- stopping chain" ; exit 1
  fi
  cur=$new
done
echo "[chain $SEED] DONE at epoch $cur"
