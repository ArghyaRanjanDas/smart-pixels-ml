#!/usr/bin/env bash
# O25 distillation chunk chain.  usage: SEED TOTAL CHUNK NFILTERS KL_BETA ARM
#
# Same chunk-chain rationale as o25_chunk_run.sh: the driver leaks ~80 MB/epoch
# in pred_stats() (a known, documented, unfixed second leak), so a long budget
# must be split into processes.  Resume is faithful: --epochs is ABSOLUTE,
# last.weights.hdf5 is written every epoch, optimizer + MDMM lambdas restore.
#
# Teacher is FROZEN and precomputed, so beta=0 is a true standalone control that
# differs from the distilled arm in exactly one term.
set -u
SEED=${1:?seed}; TOTAL=${2:-10000}; CHUNK=${3:-50}; NF=${4:-5}; BETA=${5:-1.0}; ARM=${6:?arm}; GPUMB=${7:-3000}
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
OUT=$REPO/runs/o25_distill/$ARM/s$SEED
CACHE=$REPO/runs/o25_qconv2d/_preslice_cache
TEACH=$REPO/runs/o25_qconv2d/_teacher_cache
mkdir -p "$OUT"
cur_ep () { local h; h=$(ls "$OUT"/seed_*/history.csv 2>/dev/null | head -1)
            [ -n "$h" ] && echo $(( $(wc -l < "$h") - 1 )) || echo 0; }
cur=$(cur_ep)
echo "[$ARM $SEED] start ep $cur -> $TOTAL, chunk $CHUNK, n_filters=$NF, kl_beta=$BETA"
while [ "$cur" -lt "$TOTAL" ]; do
  next=$(( cur + CHUNK )); [ "$next" -gt "$TOTAL" ] && next=$TOTAL
  env SMARTPIX_MODEL_NAME=QConv2D_Max SMARTPIX_LOSS_V2=1 SMARTPIX_PRESLICE=10,21 \
      SMARTPIX_PRESLICE_CACHE="$CACHE" SMARTPIX_TEACHER_CACHE="$TEACH" \
      SMARTPIX_KL_BETA="$BETA" SMARTPIX_NFILTERS="$NF" \
      SMARTPIX_ZEROBIAS=1 SMARTPIX_BINBAL=byterm SMARTPIX_NBINS=15 \
      SMARTPIX_BIAS_TOL=0.17 SMARTPIX_BIAS_MAXLAM=5.0 SMARTPIX_BIAS_INFCAP=2.0 \
      SMARTPIX_INIT_THR=8.565593719482422,20.227340698242188,47.49168395996094 \
      SMARTPIX_FREEZE_THR=1 SMARTPIX_ABORT_THR=1e12 SMARTPIX_GPU_MEM_MB="$GPUMB" \
      "$PY" "$REPO/scripts/distillation/run_o25_distill_kl.py" \
        --epochs "$next" --target 1 --seeds "$SEED" --out "$OUT" --extend \
        >> "$OUT/chain.log" 2>&1
  new=$(cur_ep)
  [ "$new" -le "$cur" ] && { echo "[$ARM $SEED] NO PROGRESS at $cur (wanted $next) -- stopping"; exit 1; }
  cur=$new
done
echo "[$ARM $SEED] DONE at epoch $cur"
