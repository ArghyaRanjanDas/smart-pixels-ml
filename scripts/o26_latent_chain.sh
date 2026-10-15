#!/usr/bin/env bash
# O26 (JEPA-lite) latent-distillation chunk chain.
#   usage: SEED TOTAL CHUNK NFILTERS LAMBDA ARM [GPUMB] [OUTROOT]
# Identical recipe to o25_distill_chain.sh (constrained: zero-bias + by-term bin
# balance, frozen thresholds, [10,21] readout, same chunking) -- the ONLY change
# is the distillation target: the frozen teacher's 64-dim dense_11 latent through
# a training-only predictor, instead of the forward KL on its 14 outputs. So an
# O26 arm pairs seed-for-seed with o25_distill/{wide_kl,wide_ctrl,narrow_kl}.
set -u
SEED=${1:?seed}; TOTAL=${2:-10000}; CHUNK=${3:-50}; NF=${4:-5}; LAM=${5:?lambda}; ARM=${6:?arm}
GPUMB=${7:-3000}
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
OUTROOT=${8:-$REPO/runs/o26_jepa}
PY=/work/users/das214/envs/smartpix-2bit/bin/python
OUT=$OUTROOT/$ARM/s$SEED
CACHE=$REPO/runs/o25_qconv2d/_preslice_cache
LAT=${LATDIR:-$REPO/runs/o26_jepa/_teacher_latent}
mkdir -p "$OUT"
cur_ep () { local h; h=$(ls "$OUT"/seed_*/history.csv 2>/dev/null | head -1)
            [ -n "$h" ] && echo $(( $(wc -l < "$h") - 1 )) || echo 0; }
cur=$(cur_ep)
echo "[$ARM $SEED] start ep $cur -> $TOTAL, chunk $CHUNK, n_filters=$NF, latent_lambda=$LAM"
while [ "$cur" -lt "$TOTAL" ]; do
  next=$(( cur + CHUNK )); [ "$next" -gt "$TOTAL" ] && next=$TOTAL
  env SMARTPIX_MODEL_NAME=QConv2D_Max SMARTPIX_LOSS_V2=1 SMARTPIX_PRESLICE=10,21 \
      SMARTPIX_PRESLICE_CACHE="$CACHE" SMARTPIX_LATENT_CACHE="$LAT" \
      SMARTPIX_LATENT_LAMBDA="$LAM" SMARTPIX_NFILTERS="$NF" \
      SMARTPIX_LATENT_TAP="${LTAP:-last}" SMARTPIX_LATENT_LOSS="${LLOSS:-mse}" \
      SMARTPIX_ZEROBIAS=1 SMARTPIX_BINBAL=byterm SMARTPIX_NBINS=15 \
      SMARTPIX_BIAS_TOL=0.17 SMARTPIX_BIAS_MAXLAM=5.0 SMARTPIX_BIAS_INFCAP=2.0 \
      SMARTPIX_INIT_THR=8.565593719482422,20.227340698242188,47.49168395996094 \
      SMARTPIX_FREEZE_THR=1 SMARTPIX_ABORT_THR=1e12 SMARTPIX_GPU_MEM_MB="$GPUMB" \
      "$PY" "$REPO/scripts/distillation/run_o26_latent.py" \
        --epochs "$next" --target 1 --seeds "$SEED" --out "$OUT" --extend \
        >> "$OUT/chain.log" 2>&1
  new=$(cur_ep)
  [ "$new" -le "$cur" ] && { echo "[$ARM $SEED] NO PROGRESS at $cur (wanted $next) -- stopping"; exit 1; }
  cur=$new
done
echo "[$ARM $SEED] DONE at $cur"
