#!/bin/bash -l
# NOTE the `-l`: Slurm batch scripts run in a NON-login shell where Lmod is not
# initialised and `module` does not exist.
# ---------------------------------------------------------------------------
# O25 DISTILLATION -- forward KL[teacher || student] into QConv2D.
#
#   SEED=60342 ARM=narrow_kl NF=5  BETA=1.0 sbatch scripts/gautschi/submit_o25_distill.sbatch
#   SEED=60342 ARM=wide_kl   NF=16 BETA=1.0 sbatch ...
#   SEED=60342 ARM=wide_ctrl NF=16 BETA=0   sbatch ...
#
# Teacher: runs/o22_ep100/seed_22042, precomputed to runs/o25_qconv2d/_teacher_cache.
# Chosen because its readout (slices [10,21], thr [8.57,20.23,47.49]) is IDENTICAL
# to what the students run frozen -- the project's BEST checkpoint (o24b_recover)
# uses [10,18] and would have teacher and student seeing different inputs.
# Verified softplus-dialect: relu scoring gives 1e63, softplus gives -40,137
# against a recorded best_val_loss of -40,448.
#
# BETA=0 is a true standalone control: the teacher is loaded but its term is
# dropped, so it differs from the KL arm in exactly one term.
#
# UNITS: distill.gaussian_kl returns a per-event MEAN; this lineage's NLL is a
# batch SUM. The driver multiplies the KL back up by the batch size. Without
# that the KL runs at 1/5000 weight and looks inert (measured: 13.9 out of
# 18,756), which would read as "distillation does nothing" but is a units bug.
# ---------------------------------------------------------------------------
#SBATCH --job-name=o25dist
#SBATCH -A cms
#SBATCH -p smallgpu
#SBATCH --gpus-per-node=1
# smallgpu REQUIRES 64 CPUs per GPU (memory is allocated proportional to CPUs);
# 32 is refused outright with "Batch job submission failed: Unspecified error".
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/o25d_%x_%j.out
#SBATCH --error=logs/o25d_%x_%j.err
set -euo pipefail
DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
SCRATCH=/scratch/gautschi/das214
REPO=$DEPOT/smart-pixels-ml
VENV=$SCRATCH/envs/smartpix-2bit
SEED=${SEED:?set SEED}; ARM=${ARM:?set ARM}; NF=${NF:-5}; BETA=${BETA:-1.0}
TARGET_EPOCHS=10000
OUT=$REPO/runs/o25_distill/$ARM/s$SEED
mkdir -p logs "$OUT"
module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export SMARTPIX_TFR=$SCRATCH/TFR_files_all101_noise_contained_discovery
export TF_CPP_MIN_LOG_LEVEL=1
# 42 GB on the 46 GB L40S: keep any OOM INSIDE TF (clean resume) rather than in
# the CUDA context. pred_stats() leaks ~80 MB/epoch for QConv2D, so a chunk is
# EXPECTED to die partway; resume is exact, so the chain grinds forward.
export SMARTPIX_GPU_MEM_MB=42000
export SMARTPIX_MODEL_NAME=QConv2D_Max
export SMARTPIX_LOSS_V2=1
export SMARTPIX_ABORT_THR=1e12
export SMARTPIX_PRESLICE=10,21
export SMARTPIX_PRESLICE_CACHE=$REPO/runs/o25_qconv2d/_preslice_cache
export SMARTPIX_TEACHER_CACHE=$REPO/runs/o25_qconv2d/_teacher_cache
export SMARTPIX_KL_BETA=$BETA
export SMARTPIX_NFILTERS=$NF
# readout FIXED to the teacher's, so teacher and student see identical inputs
export SMARTPIX_INIT_THR=8.565593719482422,20.227340698242188,47.49168395996094
export SMARTPIX_FREEZE_THR=1
# the O24/O25 loss stack, unchanged -- the KL is the only addition
export SMARTPIX_ZEROBIAS=1
export SMARTPIX_BINBAL=byterm
export SMARTPIX_NBINS=15
export SMARTPIX_BIAS_TOL=0.17
export SMARTPIX_BIAS_MAXLAM=5.0
export SMARTPIX_BIAS_INFCAP=2.0
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-none} | O25 DISTILL $ARM seed $SEED ==="
echo "    n_filters=$NF kl_beta=$BETA target=$TARGET_EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO"
python scripts/distillation/run_o25_distill_kl.py \
    --epochs "$TARGET_EPOCHS" --seeds "$SEED" --extend --target 1 --out "$OUT" || true
HIST=$OUT/seed_$SEED/history.csv
if [[ -f $HIST ]]; then
    LAST=$(tail -1 "$HIST" | cut -d, -f1 | tr -d '\r')
    if [[ "$LAST" =~ ^[0-9]+$ ]] && (( LAST < TARGET_EPOCHS - 1 )); then
        echo "epoch $LAST < $((TARGET_EPOCHS-1)) -- resubmitting $ARM seed $SEED"
        # RESUBMIT GUARD: sbatch can be REFUSED (QOS cap). Retry, then shout.
        for _try in 1 2 3 4 5; do
            if cd "$REPO" && SEED=$SEED ARM=$ARM NF=$NF BETA=$BETA \
                 sbatch scripts/gautschi/submit_o25_distill.sbatch; then
                echo "resubmit ok (attempt $_try)"; break
            fi
            echo "RESUBMIT FAILED (attempt $_try) -- retrying in 120s"
            if [ "$_try" = 5 ]; then echo "RESUBMIT GUARD: GIVING UP -- CHAIN DEAD"; else sleep 120; fi
        done
    else
        echo "$ARM seed $SEED reached epoch $LAST -- done"
    fi
fi
