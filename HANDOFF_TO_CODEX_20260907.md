# Handoff to Codex — 2026-09-07

Written by the Claude Code session that ran campaigns O22, O23 and O24. Assume
you know nothing about this project. Everything below is checked against the
working tree as of this date; where I state a number, it came from a run on
disk, not from memory.

**Nothing is running.** Zero training processes on the Analysis Facility A100,
zero jobs in the Gautschi Slurm queue, no background watchers alive. You are
starting from a quiet machine.

---

## 1. What this project is, and the current goal

CMS is building a **smart pixel ASIC**: a silicon pixel sensor with a small
neural network on the readout chip itself. The chip cannot afford to ship the
full waveform off-detector, so it is allowed to read only:

* **2 of 101 time slices** of the charge waveform, and
* **2 bits per pixel** (3 comparator thresholds) from a 16×16 pixel cluster.

From that, the network regresses four track parameters — `x`, `y`, `cot α`,
`cot β` — together with a full 4×4 Gaussian covariance (so downstream track
fitting gets honest per-event uncertainties, not just point estimates).

The model is a small Vision Transformer, `ViT_MaxDeep_PairLattice`, ~711K
parameters:

```
input (16,16,101) raw mV
  └─ PairLatticeRouterLayer   one logit ψ per C(101,2)=5,050 slice PAIRS.
     │                        SAMPLES one pair per training step from softmax(ψ);
     │                        takes the argmax only at evaluation.   → (16,16,2)
  └─ SoftQuantizeLayer        2 bits: 3 trainable thresholds, sharpness k
     │                        annealed 1 → 67 (cosine) during training
  └─ ViT trunk                (3,4) patches → 20 tokens → Dense(64)+pos-embed
     │                        → 4 transformer blocks (MHA 4 heads, key_dim 64,
     │                        FF 128, dropout 0.1) → LayerNorm → Flatten(1280)
  └─ deep head                Dense 256 → 128 → 64 → Dense(14) linear
```

The 14 outputs are 4 means plus the 10 lower-triangular Cholesky entries of the
covariance. Loss is `custom_loss_v2` (`two_bit_optimization_helpers/conditional_nll.py:121`):
a TFP-free 4-D Gaussian NLL, softplus diagonal, **unclipped**, summed over the
batch.

**The goal right now.** A previous campaign produced a good model (the
"parent", NLL −40,162) whose residuals are *biased*: bin the test set by the
true value of a parameter and the mean residual is not flat — it sags by up to
6.42° in α. That is regression dilution, and it matters because a systematic
tilt does not average away along a track the way random error does. Campaign
O22 removed most of that bias by adding constraints to the loss, at a cost in
resolution. **O24 asked whether training with those constraints from scratch
would do better than patching a converged model. It finished today; the answer
is no.** The best model in the project came from the opposite direction and is
described in §3.1.

---

## 2. Exact current state

### 2.1 Campaign ledger (all complete)

Naming convention: `o<N>` = experiment number, `v<k>` = revision, trailing
letter = arm, trailing digit = re-run. So `o21v2a2` is experiment 21,
revision 2, arm a, re-run 2.

| campaign | what it was | outcome |
|---|---|---|
| **O21v2a2 — "the parent"** `runs/o21v2a2_pairlattice/seed_22042/` | cold discovery of slice pair + thresholds, 10,000 epochs, no bias constraint | slices **(10,21)**, thresholds **[8.57, 20.23, 47.49] mV**, plain val NLL **−40,162**. Its −40K came from a **phase transition between epoch 2,000 and 3,000** (train/val gap *collapsed* +4,680 → +1,679 — a generalization event, not just more fitting). Only ~1 seed in 5 reached that basin. |
| **O22 — bias removal** `runs/o22_ep100/seed_22042/` | warm start from the parent, slice pair and thresholds **frozen**, 100 epochs with two new loss terms | **This is the currently shipped result.** α bias 6.42° → 2.35°, x 3.60 → 1.39 µm, y 1.14 → 0.51 µm, for σx +0.2% / σy +1.0% / σα +16.8% / σβ +16.1%. |
| **O23 — information ceiling** `runs/o23_*` | is −40K the network's limit or the readout's? Train router-less arms that see 20 or 101 slices at full float precision | **Inconclusive — all arms overfit.** Without the router's per-step slice sampling the model memorises the 155k training events. Best −24.1K to −27.5K. Not a ceiling measurement. Arm B (2 slices, float) was never run and is still the cleanest unanswered question here. |
| **O24 cold** `runs/o24_scratch_bias/s{40342,40442,40542,40642}/` | the O22 loss from epoch 0 with the parent's full recipe (free router, trainable thresholds, coarse-to-fine smoothing) | 4 seeds × 10,000 epochs, all finished. **No seed reproduced the parent's transition.** They match O22 on bias but pay σx +34–46% and σy +65–106%. |
| **O24b warm** `runs/o24b_from_o22/s{41042,41142,41242}/` | warm start from the O22 checkpoint, router ψ **inherited and trainable**, thresholds free, 10,000 epochs | 3 seeds, all finished. The router migrated to **anchor 8** in every seed and thresholds fell ~35–40%. That readout moves resolution cost from angles onto positions. Every seed's best plain NLL occurred in its **first ~6 epochs** and was never beaten in the remaining ~9,990. |
| **O24b recovery** `runs/o24b_recover/s41142/seed_41142/` | 12 epochs from the O22 checkpoint with `best.weights` selected on plain NLL instead of the composite loss | **Produced the best model in the project.** See §3.1. |

### 2.2 The numbers that matter

Full test set, 37,919 events, `val_plain_nll` (the only NLL comparable across
arms — see §5.4). Bias is `max |mean(true − pred)|` over 15 truth bins.
Δσ is against the parent. Figures: `runs/perf_plots_o24_final/bias_comparison.png`.

| model | readout | thresholds mV | x bias | y bias | α bias | β bias | Δσx | Δσy | Δσα | Δσβ | plain NLL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| parent | (10,21) | 8.57, 20.23, 47.49 | 3.60 | 1.14 | 6.42 | 1.62 | — | — | — | — | −40,162 |
| O22 ep100 (shipped) | (10,21) | frozen | 1.39 | 0.51 | 2.35 | 0.85 | +0.2% | +1.0% | +16.8% | +16.1% | −39,775 |
| **recovered** ← best | (10,21) | 8.50, 20.13, 47.32 | 1.68 | 0.50 | 2.58 | 0.63 | **−1.1%** | **−0.2%** | **+12.9%** | **+11.2%** | **−40,337** |
| cold 40342 | (10,19) | 5.68, 16.95, 39.82 | 1.75 | 0.57 | 1.88 | 0.58 | +34.4% | +65.2% | +24.2% | +29.5% | −38,058 |
| cold 40442 | (8,20) | 4.97, 16.39, 39.24 | 1.08 | 0.58 | 2.83 | 0.59 | +45.9% | +85.2% | +21.3% | +30.6% | −37,513 |
| cold 40542 | (11,20) | 5.71, 16.91, 40.14 | 1.24 | 0.56 | 1.99 | 1.01 | +38.4% | +74.5% | +62.0% | +70.0% | −37,720 |
| cold 40642 | (8,19) | 5.39, 16.49, 40.19 | 1.22 | 0.48 | 1.48 | 0.60 | +41.2% | +106.2% | +18.1% | +23.4% | −37,907 |
| warm 41142 (final) | (8,21) | 5.25, 15.60, 38.94 | 1.42 | 0.34 | 2.84 | 0.56 | +18.2% | +40.8% | +16.2% | +21.1% | −37,874 |
| warm 41242 (final) | (8,19) | 5.42, 16.09, 39.51 | 1.55 | 0.33 | 1.68 | 0.66 | +9.0% | +18.2% | +14.8% | +19.8% | −38,457 |

Pull widths (residual / predicted σ, Gaussian fit) for every model lie between
**0.83 and 1.06**. That is the check that makes the Δσ column trustworthy: no
model is buying a flat residual line by inflating its own error bars, so the
resolution differences are real information, not bookkeeping. Per-seed pull and
summary plots: `runs/perf_plots_o24_seeds/` (20 PNGs, `summary_o22_<arm>.png`
and `pull_o22_<arm>.png`).

### 2.3 What is broken / imperfect

* **`git status` shows one modified file:** `scripts/distillation/make_perf_plots_o22.py`.
  This is a *bug fix I made and verified but was told not to commit*. Line 97
  now reads `_e = h[1]; xdata = _e[:-1] + (_e[1] - _e[0]) / 2.0`. It previously
  read `xdata = h[1][:-1] + 3/50.` — a constant left over from an older ±3 pull
  range. With the current `np.linspace(-5,5,50)` the bins are width 10/49 =
  0.2041, so the correct centre offset is 0.1020, not 0.06. **Every pull µ
  printed before today was 0.042 too low.** σ was unaffected (a uniform x-shift
  moves the fitted mean, not the width). All plots in
  `runs/perf_plots_o24_seeds/` were regenerated with the fix; anything in
  `runs/perf_plots_o22/` and older directories still has the old µ. Commit this.
* **`runs/o24b_from_o22/s41042/` has no per-bin bias evaluation.** Its σ and
  pull numbers exist (`runs/perf_plots_o24_seeds/`), its bias numbers do not.
  Nobody ran it through `make_bias_comparison.py`.
* **Three archived failure directories** are kept deliberately, each with a
  `README.txt` explaining what went wrong: `runs/_o22_onehot_overfit_{on,off}/`,
  `runs/_o22_runaway_notol_on/`, `runs/_o24_wrongarch_cold/`,
  `runs/_o24b_smoothtrap/`. **Do not delete them** — they are the evidence for
  why the current configuration looks the way it does.

---

## 3. Immediate next steps, in priority order

### 3.1 Reproduce the winning checkpoint on two more seeds  ← highest value

The best model in the project is
`runs/o24b_recover/s41142/seed_41142/best.weights.hdf5`. It is the O22 ep-100
checkpoint plus **2 epochs** of continued constrained training, with
`best.weights` chosen on plain NLL. It **dominates the shipped O22 result on
every axis**: same bias, a quarter less angular-resolution cost, position
resolution slightly better than the parent's, and the best likelihood any model
in this project has produced (−40,337 vs the parent's −40,162).

It is **one seed**. Before it goes in a paper, run the identical 12-epoch job on
the other two warm seeds and check the numbers hold:

```bash
cd /work/users/das214/SmartPixels/smart-pixels-ml
R=$PWD
for SEED in 41042 41242; do
  SMARTPIX_MODEL_NAME=ViT_MaxDeep_PairLattice SMARTPIX_LOSS_V2=1 \
  SMARTPIX_ZEROBIAS=1 SMARTPIX_BINBAL=byterm SMARTPIX_NBINS=15 \
  SMARTPIX_BIAS_TOL=0.17 SMARTPIX_BIAS_MAXLAM=5.0 SMARTPIX_BIAS_INFCAP=2.0 \
  SMARTPIX_WARM_START=$R/runs/o22_ep100/seed_22042/last.weights.hdf5 \
  SMARTPIX_WARM_SRC_MODEL=ViT_MaxDeep_PairLattice \
  SMARTPIX_PIN_MODE=inherit_free SMARTPIX_FIX_SLICES=10,21 \
  SMARTPIX_CKPT_MONITOR=val_plain_nll SMARTPIX_ABORT_THR=1e12 \
  setsid nohup /work/users/das214/envs/smartpix-2bit/bin/python \
    scripts/distillation/run_simplerouter_mdmm_discovery.py \
    --epochs 12 --target 1 --seeds $SEED \
    --out runs/o24b_recover/s$SEED > /tmp/recover_$SEED.log 2>&1 < /dev/null &
  sleep 20
done
```

Each takes ~5 minutes on the A100. Then evaluate all three against the parent
and the shipped O22 point (command in §6.3).

### 3.2 Evaluate warm seed 41042's bias

The only gap in the results table. Add
`warm_41042_final:$R/runs/o24b_from_o22/s41042/seed_41042/last.weights.hdf5`
to a `SMARTPIX_ARMS` list and run `make_bias_comparison.py` (§6.3). CPU only,
~3 minutes.

### 3.3 Commit the pull-fit bug fix

`scripts/distillation/make_perf_plots_o22.py` line 97. See §2.3. I was
explicitly told not to commit during this handoff, so it is sitting in the
working tree.

### 3.4 Write O24 into the results artifact

There is a published web report covering O22 and O23 at
`https://claude.ai/code/artifact/61401283-d679-4dd3-8fe4-ad2d2f0961f4`, and a
per-seed slide deck for O24 at
`https://claude.ai/code/artifact/c03c5e7d-65cf-4d79-bd1f-78efca4b6049`.
**Both are Claude-hosted and you cannot update them.** The underlying figures
are all in `runs/perf_plots_*`; the narrative is in
`docs/handoff/HANDOFF_2026-09-07.md` §10. If the user wants the O24 story in a
shareable document, produce a new one from the figures rather than trying to
edit those URLs.

### 3.5 Optional: O23 arm B

The one clean experiment nobody ran. `scripts/distillation/run_ceiling_arm.py
--arm B` trains 2 slices at **full float precision** — same slice budget as the
parent, so the same memorisation pressure that made arms A and C overfit should
not apply. It isolates what the 2-bit quantisation alone costs, which is
directly the number a chip designer needs when deciding whether a 4th
comparator is worth the silicon.

---

## 4. Decisions already made, and why — do not silently undo these

1. **`best.weights` is now selectable via `SMARTPIX_CKPT_MONITOR`**
   (`run_simplerouter_mdmm_discovery.py:943`), default `val_loss`. Under the
   bin-balanced loss, `val_loss` is a *weighted composite* whose minimum is not
   the likelihood minimum. That is precisely how the −40.4K state was lost in
   O24b: it occurred at epoch 2–6, the composite kept improving later, and the
   plain-NLL optimum was never written to disk. **For any run where you care
   about likelihood, set `SMARTPIX_CKPT_MONITOR=val_plain_nll`.**

2. **`PIN_MODE=inherit_free` exists and is not redundant**
   (`run_simplerouter_mdmm_discovery.py:722`). There are three modes:
   `onehot` spikes ψ at one pair and freezes it (this **destroys the router's
   per-step sampling**, which is load-bearing regularisation — it caused a
   documented overfit, see `runs/_o22_onehot_overfit_on/`); `inherit` copies ψ
   from the source checkpoint and freezes it (what O22 used, to attribute its
   result to the objective alone); `inherit_free` copies ψ and **leaves it
   trainable** (what O24b needed, to ask whether the readout moves). The plain
   warm-start path deliberately does *not* transfer the router at all and
   asserts ψ was untouched.

3. **The zero-bias constraint has a tolerance band and hard caps**, not an
   equality. `inf = max(0, |bias|/σ − 0.17)`, `max_lambda=5`, `inf_cap=2`
   (`mdmm.py:123` onwards). An equality constraint on a per-batch-noisy
   statistic makes λ integrate noise forever; that produced a documented
   runaway (`runs/_o22_runaway_notol_on/`, training loss reached +354,848).
   **The 0.17 is derived, not tuned**: per-batch bin-mean noise is
   5000 events / 15 bins ≈ 356 per bin → 0.056σ; three times that is 0.17. An
   earlier derivation used the *evaluation* set's noise (0.021σ → tol 0.06) and
   was three times too tight.

4. **σ (the constraint's denominator) is `stop_gradient`'d.** Without it the
   model can satisfy "bias/σ is small" by inflating σ instead of reducing bias.

5. **The O24 cold campaign ran its full 10,000-epoch budget deliberately.**
   The user's explicit instruction, on the reasoning that this system produces
   late jumps — the parent improved at epoch ~4,999 after a 768-epoch stale
   stretch. It was the right call even though no seed transitioned: three of
   four cold seeds were still setting records past epoch 8,900, one as late as
   **9,959**. Do not offer to stop a user-set budget early.

6. **`runs/perf_plots_o22/` is not the output directory for new evaluations.**
   Both plotting scripts now honour `SMARTPIX_OUT`
   (`make_bias_comparison.py:38`, `make_perf_plots_o22.py:33`) so evaluations
   stop overwriting the O22 figures. Always set it.

---

## 5. Gotchas and traps

### 5.1 `SMARTPIX_MODEL_NAME` is effectively mandatory

`run_simplerouter_mdmm_discovery.py:647` falls back to
`ViT_Max_SimpleRouter` — a **shallow-head, 1-D router** model — if
`SMARTPIX_MODEL_NAME` is unset. It fails silently: the run trains happily and
produces plausible-looking losses for the wrong architecture. This cost a full
campaign; the evidence is archived in `runs/_o24_wrongarch_cold/`. **Always set
`SMARTPIX_MODEL_NAME=ViT_MaxDeep_PairLattice`.** The Gautschi sbatch does it
(`scripts/gautschi/submit_o24_scratch_bias.sbatch:50`); hand-rolled command
lines are where it gets forgotten.

### 5.2 Never copy the smoothing schedule into a warm start

`SMARTPIX_SMOOTH_SIGMA0=4 SMARTPIX_SMOOTH_EPOCHS=3000` is a coarse-to-fine
kernel over the 5,050-pair lattice. It exists to help a **blank** router
explore. Applied to an inherited ψ it resets the kernel to σ=4 at epoch 0 and
**blurs the incumbent pair away instantly** — the router then collapses into
the adjacency trap ((13,14)/(14,15) type pairs). Evidence:
`runs/_o24b_smoothtrap/`. The nasty part is that the driver's own log prints
"sampler draws the incumbent 3.7% of steps" *before* the scheduler runs, so the
log looks correct and is not. **Warm starts must omit both SMOOTH variables**
so the source's σ≈0 carries over. Verify by decoding ψ from
`phi_history.npz` at epoch 0 and checking the intended pair leads.

### 5.3 `--target N` counts already-converged seeds in the SAME output directory

If several seeds share one `--out`, the first to converge satisfies `--target 1`
and every other chain **exits immediately without training and without
complaining**. Always use **one output directory per seed**:
`--out runs/<campaign>/s<SEED>`.

### 5.4 NLL comparability

Three different loss "dialects" exist in this repo's history. Only compare
`val_plain_nll`, the unweighted `custom_loss_v2` value logged as a metric. The
training `loss` column under bin balancing is a weighted composite. The older
O11–O21 ledger used a clipped v1 loss and is **not** comparable to anything
current. Also: the value is a batch-SUM averaged over batches, so if you compute
it yourself, divide by the **number of batches**, not by events/5000 — the last
batch is short and dividing by events gives a number ~5% off.

### 5.5 The AF pod's `/proc/loadavg` and `free` are the *host's*, not yours

This wasted hours. Inside the Purdue AF pod, `uptime` reports the whole Geddes
node (shared with other tenants' pods) — a load average of ~390 on a 128-core
session is normal and says nothing about your jobs. The authoritative numbers:

```bash
cat /sys/fs/cgroup/memory/memory.usage_in_bytes   # your pod's real usage
cat /sys/fs/cgroup/memory/memory.limit_in_bytes   # 256 GiB
cat /sys/fs/cgroup/cpu/cpu.stat                   # nr_throttled — has always been 0
```

The tell for a host-wide reading: the 1/5/15-minute figures sit at the same
value for hours. Real contention fluctuates.

### 5.6 Environment packages that are not in the environment

`/work/users/das214/envs/smartpix-2bit` (Python 3.10, numpy 1.23.5, TF 2.15.1)
was silently relying on an inherited `PYTHONPATH` for two packages. When that
went away, everything broke at import:

* **`networkx`** — `qkeras` imports it at module load, so *every* driver run
  dies before doing anything.
* **`scipy`** — `curve_fit`, needed for the pull-plot Gaussian fits.

Both are now installed into the env. **scipy was pinned with `--no-deps` to
1.13.1 on purpose.** Do not let pip upgrade numpy in this env — TF 2.15 is built
against numpy 1.23 and an upgrade will break it in confusing ways.

### 5.7 `setsid nohup … &` and *how to wait for it*

Long jobs must be launched with `setsid`, not bare `nohup`: a tool-call timeout
kills a plain detached child. But `setsid` forks, so **the PID you capture with
`$!` exits immediately** and any `wait`/`until ps -p $!` loop fires instantly.
Wait on the **log file's contents**, never the PID.

Related, and it bit me today: `while ps -eo cmd | grep -q "..."` inside a script
with `set -o pipefail` exits on its first iteration — `grep -q` closes the pipe,
the upstream `grep` dies of SIGPIPE, the pipeline "fails". That launched a fifth
trainer onto a GPU sized for four. Use `pgrep`, or an `awk` match on
`ps -eo pid,args` output.

### 5.8 `pkill -f <script>` matches your own shell

Killing by pattern matches the calling `bash -c` whose command line contains the
script name, so you kill yourself mid-sequence. Select PIDs by exact executable
path:

```bash
ps -eo pid,args --no-headers | awk '$2=="/work/users/das214/envs/smartpix-2bit/bin/python" && index($0,"o24b_recover")>0 {print $1}'
```

### 5.9 Stale `git index.lock`

This repo has repeatedly ended up with a stale `.git/index.lock` after an
interrupted command, silently blocking every commit. Before removing one,
confirm there is genuinely no git process:
`ps -eo args --no-headers | awk '$1=="git" || $1 ~ /\/git$/'`. Two locks once
sat there for months.

### 5.10 The A100 fits 3–4 trainers, not more

Nine concurrent trainers OOM'd. Four is the working ceiling for this model at
batch 5000.

### 5.11 Torn CSV lines parse into plausible numbers

`history.csv` rows written during a crash have produced values like −45,408
among neighbours at −39K, and once a `val_loss` of −37,265,446 that permanently
poisoned Keras's best-weights monitor (a checkpoint froze for 4,400 epochs).
When reading histories programmatically, filter on an **absolute** plausibility
bound (`abs(v) < 5e6`), not a local-median cut — a local cut throws away real
early spikes.

### 5.12 Paths that are symlinks

`/depot/cms/users` → `/depot/cms/private/users`. So
`~/depot/users/das214`, `/depot/cms/users/das214` and
`/depot/cms/private/users/das214` are all one directory. Stack traces will show
you the `private` form even when you referenced the other.

---

## 6. How to run things

All commands assume `cd /work/users/das214/SmartPixels/smart-pixels-ml`.
The Python interpreter is `/work/users/das214/envs/smartpix-2bit/bin/python`
(**not** whatever `python` resolves to).

### 6.1 Live status of everything, on both machines

```bash
bash scripts/status_o24.sh
```
Needs no session state; reads process lists, `history.csv` on the AF, the Depot
copies of the Gautschi runs, and `squeue` over SSH.

### 6.2 Train (the campaign driver)

```bash
SMARTPIX_MODEL_NAME=ViT_MaxDeep_PairLattice \   # MANDATORY, see §5.1
SMARTPIX_LOSS_V2=1 \
SMARTPIX_SMOOTH_SIGMA0=4 SMARTPIX_SMOOTH_EPOCHS=3000 \   # COLD runs only, see §5.2
SMARTPIX_ZEROBIAS=1 SMARTPIX_BINBAL=byterm SMARTPIX_NBINS=15 \
SMARTPIX_BIAS_TOL=0.17 SMARTPIX_BIAS_MAXLAM=5.0 SMARTPIX_BIAS_INFCAP=2.0 \
SMARTPIX_ABORT_THR=1e12 \
setsid nohup /work/users/das214/envs/smartpix-2bit/bin/python \
  scripts/distillation/run_simplerouter_mdmm_discovery.py \
  --epochs 10000 --target 1 --seeds <SEED> --extend \
  --out runs/<campaign>/s<SEED> > /tmp/<name>.log 2>&1 < /dev/null &
```

Warm start instead adds, and omits both SMOOTH variables:

```bash
SMARTPIX_WARM_START=$PWD/runs/o22_ep100/seed_22042/last.weights.hdf5 \
SMARTPIX_WARM_SRC_MODEL=ViT_MaxDeep_PairLattice \
SMARTPIX_PIN_MODE=inherit_free SMARTPIX_FIX_SLICES=10,21
```

`--extend` resumes from `last.weights.hdf5` + `opt_state` in the output
directory; it prints `RESUMING from epoch N`. **Check for that line** — a silent
fresh start looks identical otherwise.

Full env-var list with line numbers: `SMARTPIX_BATCH:135`,
`WARM_START:140`, `WARM_SRC_MODEL:141`, `ZEROBIAS:158`, `BINBAL:159`,
`NBINS:160`, `BIAS_SCALE:166`, `BIAS_TOL:170`, `BIAS_START:173`,
`BIAS_MAXLAM:174`, `BIAS_INFCAP:175`, `PIN_MODE:189`, `SMOOTH_SIGMA0:210`,
`MODEL_NAME:213`, `SMOOTH_EPOCHS:214`, `FIX_SLICES:224`,
`CKPT_MONITOR:943` — all in `scripts/distillation/run_simplerouter_mdmm_discovery.py`.

### 6.3 Evaluate checkpoints (bias + resolution trade table)

CPU only. Always set `SMARTPIX_OUT`.

```bash
R=$PWD
ARMS="parent:$R/runs/o21v2a2_pairlattice/seed_22042/last.weights.hdf5"
ARMS="$ARMS,o22_ep100:$R/runs/o22_ep100/seed_22042/last.weights.hdf5"
ARMS="$ARMS,recovered:$R/runs/o24b_recover/s41142/seed_41142/best.weights.hdf5"
CUDA_VISIBLE_DEVICES='' SMARTPIX_OUT=$R/runs/perf_plots_<name> SMARTPIX_ARMS="$ARMS" \
  /work/users/das214/envs/smartpix-2bit/bin/python scripts/distillation/make_bias_comparison.py
```

Same `SMARTPIX_ARMS` / `SMARTPIX_OUT` interface produces the per-model summary
and pull plots:

```bash
… /work/users/das214/envs/smartpix-2bit/bin/python scripts/distillation/make_perf_plots_o22.py
```

Each arm takes ~1–2 minutes (inference over 37,919 events on CPU).

### 6.4 Submit to Gautschi (Slurm)

```bash
ssh gautschi
cd /depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml
SEED=40642 sbatch scripts/gautschi/submit_o24_scratch_bias.sbatch
```

That script is a **self-resubmitting chunk chain**: the GPU there leaks
~345 MB/epoch with the bias constraint, so a chunk is *expected* to die of OOM
after ~149 epochs, and the script re-submits itself (with a 5-attempt retry
guard, because `sbatch` can be refused by the QOS job cap and a silent failure
once left seeds dead for hours). Resume is exact. `SMARTPIX_GPU_MEM_MB=42000`
is set because the nodes are **L40S with 46 GB** — the historical 20 GB value
was an A100-slice-era leftover and gave 58-epoch chunks.

**Do not run `scancel -u $USER -n o24bias` when you mean to cancel one seed** —
it cancels all four chains. (I did this; one seed had to be resubmitted.)

### 6.5 No test suite

There is none. The closest things are `scripts/distillation/smoke_o22_bias.py`
and `smoke_o22_wiring.py`, which assert the constraint wiring on a real model.
Run them after touching `mdmm.py` or `conditional_nll.py`.

---

## 7. Where the data lives

| what | path | notes |
|---|---|---|
| **training/test dataset** | `/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/TFR_files_all101_noise_contained_discovery/` | 19 GB. `TFR_train/` 31 shards × 5,000 = **155,000 events**; `TFR_test/` 8 shards = **37,919 events**. Noise is already baked in, so the loader is called with `noise=-1`. |
| same, Gautschi copy | `/scratch/gautschi/das214/TFR_files_all101_noise_contained_discovery/` | already staged; `/work` is **not** mounted on Gautschi |
| AF repo | `/work/users/das214/SmartPixels/smart-pixels-ml` | this one |
| Gautschi repo | `/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml` | `/depot` is mounted on **both** machines, so run directories can be copied directly — no scp needed |
| environment (AF) | `/work/users/das214/envs/smartpix-2bit` | see §5.6 |
| environment (Gautschi) | `/scratch/gautschi/das214/envs/smartpix-2bit` | |

**No proxy or token is needed for anything in the current workflow.** All data
is on local POSIX mounts. There is no xrootd/EOS access, no VOMS proxy, no Rucio
in this project's pipeline. (The wider Purdue AF does need `voms-proxy-init` for
remote CMS data — irrelevant here.) The one credential in play is SSH to
Gautschi, which works from this pod with `ssh gautschi` using existing keys.

A separate dataset was shared with a collaborator and lives at
`/work/projects/SmartPixML/datasets_16x16x20_charge/` (12 GB, world-readable) —
20 time frames, charge units, from arXiv 2602.15946. It is **not** used by any
code in this repo.

---

## 8. Things that must NOT be done

1. **Do not push to `origin`.** `origin` is a collaborator's repository
   (`github.com/davidgjiang/smart-pixels-ml`). Push only to `myfork`
   (`github.com/ArghyaRanjanDas/smart-pixels-ml`), branch `softtimerouter`.
   Pushing to `myfork` is pre-authorised; `origin` needs per-push approval.
2. **Do not subset a dataset** without explicit permission. Full statistics by
   default — this is a standing instruction from the user.
3. **Do not stop a user-set epoch budget early**, and do not offer opinions
   mid-run about stopping. State a risk once, then let it run. See §4.5.
4. **Do not delete the `runs/_*` archived failure directories.**
5. **Do not write reports into `/work/projects/SmartPixML`.** Reports and
   figures stay in this repo.
6. **Do not let pip upgrade numpy** in the smartpix-2bit env (§5.6).
7. **Do not run more than 4 trainers on the A100** (§5.10).
8. **Do not change an experiment's *meaning* without asking.** Operational
   decisions (restarting, resubmitting, killing a demonstrably broken run) are
   yours to make without asking; changing what an experiment measures is not.
9. **Do not trust `runs/perf_plots_o22/` pull µ values** or those in older
   perf-plot directories — they carry the 0.042 offset from §2.3.

---

## 9. Facts that exist only in conversation history

These are not in the code, the commit log, or any README. They are the reason
several things look the way they do.

**9.1 The router's sampling is load-bearing regularisation, and its winning
pair is rare.** `PairLatticeRouterLayer` samples one pair per training step from
softmax(ψ); at the parent's converged state the winning pair carries only
**p = 0.037** of the mass, so training saw its own answer on 3.7% of steps and
saw *other* pairs 96% of the time. That accidental augmentation is what keeps
this model from memorising 155k events. Every attempt to remove it — the
one-hot pin in O22, the router-less arms in O23 — overfit. If you are ever
tempted to "clean up" the sampler into a deterministic argmax, this is why you
must not.

**9.2 The parent's phase transition was a generalization event, not more
fitting.** Between epochs 2,000 and 3,000 its validation improved by 7,776 while
training improved by only 4,775, and the train/val gap *collapsed* from +4,680
to +1,679. That asymmetry is the signature to look for if you are ever judging
whether a run is about to transition. A run whose gap is *diverging* is moving
away from that kind of solution — which is how I judged the O23 10K run dead at
epoch 2,001 (gap +14,279 and tripling) rather than waiting out its budget.

**9.3 The parent improved on only 140 of its 10,000 epochs**, with stale
stretches of 523 and 768 epochs *after* it had already reached −40K. This model
does not descend smoothly; it sits on plateaus and occasionally finds a new
basin. Do not read a few hundred stale epochs as convergence.

**9.4 Two independent campaigns, from opposite directions, both moved the
thresholds down 35–40%.** O24b (warm, router free) and O24 cold (from scratch)
independently landed on thresholds around [5.0–5.8, 15.6–17.0, 38.9–40.2] mV
versus the parent's [8.57, 20.23, 47.49], and both pulled the slice pair
tighter. That readout family is *better for angles and worse for positions*.
This is a real, reproducible property of the bias-aware objective and is
probably the most interesting physics result of O24 — it is in the numbers but
nobody has written it up as a finding.

**9.5 The bias constraint makes training dramatically more reproducible.** The
four cold seeds finished within **526 NLL** of one another. The parent campaign,
same architecture without the constraint, reached its deep basin in roughly
**1 seed out of 5**. If seed-to-seed variance is ever a problem for you, this
is a lever.

**9.6 O22's warm start was itself a warm-up.** The parent trained 10,000
unconstrained epochs before the constraint was ever applied. So "constraint from
epoch 0" (O24 cold) is not the same experiment run longer — it is a
fundamentally harder optimisation. A first attempt at it saturated the
constraint completely (penalty pinned at its 900,000 ceiling against an NLL of
40,411, i.e. 97% of the gradient fighting something a random model cannot
satisfy). That attempt was on the *wrong architecture* (§5.1); the correct
architecture does **not** saturate, which is why the campaign was re-run rather
than abandoned. `SMARTPIX_BIAS_START` (a gate that holds the constraint off
until epoch N) was built as insurance and is **unused** — the correct
architecture never needed it.

**9.7 O24b's best NLL happened in the first six epochs of a 10,000-epoch run,
in all three seeds.** Everything after that made the models worse. That is what
motivated §3.1, and it is a strong hint that for this objective, "warm start and
continue briefly" beats "train longer" quite generally.

**9.8 The A100 is shared with other agent sessions working for the same user.**
A scouting HH→bbττ analysis session and a Kaggriculture session run on this same
pod. During this work the pod hit ~6× its memory request; the cause turned out
to be **9,781 orphaned `t6.py` processes** from the Kaggriculture session
(a `ProcessPoolExecutor` fan-out whose parent died without reaping its pool,
re-parented to PID 1, idle for 13.5 days, holding ~80 GB and 9.8K PIDs). They
were killed with the user's approval. If pod memory looks impossible again,
check for orphans (`ps -eo pid,ppid,args | awk '$2==1'`) before blaming your own
jobs. Other sessions are reachable by name and will coordinate — they are
generally cooperative about GPU scheduling.

**9.9 A FastML26 hackathon has intermittent priority on this A100.** A teacher
training job writes
`/work/users/das214/fastml26/fastml26-c1/team/teacher/DONE` when finished; that
file is the agreed gate. The user paused this project's training once for it.
Check `nvidia-smi` is not readable inside the pod (permission denied) — you
cannot see other tenants' GPU memory directly, so coordination is by message,
not by inspection.

**9.10 The user's working style, learned the hard way.** They want operational
decisiveness (restart, resubmit, kill broken runs — just do it) and consultation
on meaning (what an experiment measures). They ask "what is the status update?"
frequently and want a concrete answer, not a narration of process. They reason
explicitly about **grokking** — models jumping to a global minimum at high epoch
counts — which is why full budgets are non-negotiable. Being told mid-run that
something looks unlikely to improve is not welcome; I was corrected for it once
("Did I ask your opinion? Run them for 10K epochs, let's see"), and the parent's
own epoch-4,999 improvement shows they were right to insist.

---

## 10. Orientation files already in the repo

* `AGENTS.md` (repo root) — the agent router. Codex reads this automatically.
  Identical content in `CLAUDE.md`.
* `docs/handoff/HANDOFF_2026-09-07.md` — the durable project handoff written
  before this one; §10 has the O24 verdict, §9 the recovered-checkpoint result.
* `docs/agent-memory/` — 26 single-fact files ported from the previous agent's
  memory (user preferences, project facts, references), with `README.md` as the
  index.
* `scripts/status_o24.sh` — live status, no session state required.
* `scripts/o24_followon_watch.sh` — the shell watcher that ran the last two GPU
  steps autonomously. **It has already completed and is not running.** Kept as a
  worked example of a session-independent follow-on.
