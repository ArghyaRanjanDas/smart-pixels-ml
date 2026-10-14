# Handoff back to Claude — 2026-09-14

This is the update from the Codex session that followed the original Claude
campaign session. The user is returning to their research group's Claude team
subscription and expects to continue with their earlier Claude context.

**Read this alongside `HANDOFF_TO_CODEX_20260907.md`, not instead of it.** That
document remains the detailed O21–O24 campaign history. This document records
what happened afterward, corrections to several interpretations, the actual
slide artifacts, and the next conversation the user wants to have.

Repository: `/work/users/das214/SmartPixels/smart-pixels-ml`.
Verification date: **2026-09-14 UTC**. Live facts below are
snapshots, not promises about what is running when you resume.

## 1. The user's current request and the immediate next step

The meeting went well. The user now wants to explore training the **quantized
Conv2D model, QConv2D**, and explicitly asked to be challenged with questions so
we reach a shared understanding before deciding the experiment. That discussion
was interrupted by this migration request. **No new QConv2D training was launched
by this Codex session, and no new QConv2D protocol was agreed.**

The user's final clarification was to write the handoff and a prompt for Claude;
Claude can take over afterward. Do not interpret the old O24 command blocks or
the existence of a legacy QConv script as approval to launch a new campaign.
The correct next interaction is the focused QConv discussion in section 8.

The historical recovery replication remains unfinished, but the user's latest
direction is QConv2D. Do not silently put them back on an old priority list.

Ready-to-paste continuation prompt: `PROMPT_FOR_CLAUDE_20260914.md`.

## 2. What this Codex session actually completed

1. Read the original handoffs and audited the O24 results, loss definitions,
   model quantization claims, checkpoint identities and plot conventions.
2. Located the **existing ATrain lattice-animation deck**, rather than creating
   an unrelated replacement deck. Added 18 campaign slides: full training
   histories, per-checkpoint summaries and pulls, comparison, bias trade and
   conclusions. Reused the established 20 summary/pull figures.
3. Added two native-equation slides immediately before “Bias, resolution and
   pull width answer different questions,” explaining the original and current
   loss, including the inherited anti-collapse correlation constraint.
4. In response to Lindsey Gray's questions, evaluated the exact cold seed 40442
   checkpoint on all 37,919 test events, cached predictions, and added two
   residual-distribution slides explaining global standard deviation versus
   central containment and predicted uncertainty.
5. Found the requested `from_weights.ipynb`, adapted its linear filled-histogram
   method with the current dataset's units, and added two more residual slides.
6. Clarified that the central 68% interval is calculated **unbinned**, directly
   from event residuals, and is a containment interval, not a confidence interval.

These were analysis, figure and presentation tasks. This session did not train
new O24 or QConv models, replicate the recovery on new seeds, commit changes,
push to a remote, or send Slack messages to Lindsey. Suggested replies were
provided to the user to send themselves.

## 3. Scientific corrections and distinctions to retain

### Model and hardware claims

All O21 parent/O22/O24 comparisons in these slides use
`ViT_MaxDeep_PairLattice`, which the user calls **large ViT_D**, the Vision
Transformer with the deeper decoder/head. It is a floating-point network with
hard 2-bit input digitization and two selected time slices. It is **not** the
internally quantized QConv2D model and has no demonstrated ASIC fit.

The router receives the 101-slice waveform and passes one selected pair to the
backbone; training samples pairs and inference chooses the argmax. The
SoftQuantize layer has hard-forward digitization, a soft-gradient training path,
and four levels. Calling it “SoftDigitization” does not mean inputs remain
continuous in the forward pass. Internal Dense/attention/LayerNorm layers of
this ViT remain floating point.

Sources: `two_bit_optimization_helpers/models/transformer_model_nonquantized.py`,
`SoftQuantizeLayer.py`, `PairLatticeRouterLayer.py` in that helpers directory,
and the factory in `two_bit_optimization_helpers/train.py`.

### Bias, resolution, uncertainty and pulls are different statistics

- Residual: **true minus predicted**. Positions are reported in micrometres;
  angular residuals are calculated after converting cotangents to degrees.
- Table σ: **population standard deviation about the global residual mean**,
  `np.std(residuals, ddof=0)`, across the entire test set. It includes tails and
  truth-dependent bias. It is not a Gaussian fit, a FWHM, a per-bin width, or
  the RMS about zero.
- Central 68% containment: `[q16, q84]` from
  `np.quantile(residuals, [0.16, 0.84])` on all individual residuals. Its quoted
  half-width is `(q84 - q16) / 2`. This calculation is **unbinned**, independent
  of displayed histogram binning, and not a shortest-interval construction.
- Blue summary band: average **predicted marginal uncertainty in each truth
  bin**, placed around that bin's mean residual. For angles the code transforms
  cotangent ± uncertainty endpoints to angular offsets. This is not the
  empirical global σ, central 68% interval, or standard error on the mean.
- Dots/error bars in the existing summaries use binned means and seaborn's
  default bootstrap confidence intervals. These are different again.
- Pulls use residual / predicted marginal σ in x, y, **cot α, cot β**. Their
  quoted widths are Gaussian core fits over the plotted range. A core width
  near one does **not** demonstrate tail calibration or full-covariance
  calibration. Do not inherit the original handoff's stronger calibration claim.

There is no confirmed angular upper/lower-band swap bug: inverse cotangent is
decreasing, but the residual is true minus predicted, which reverses the sign
again. That apparent concern was checked; do not report it as a discovered bug.

### Interpretation of the recovery and campaign trends

The recovered checkpoint improves all four global residual widths and plain
NLL relative to O22. It does **not strictly dominate every bias metric**:
maximum x bias is about 1.68 versus 1.39 µm, and α bias 2.58 versus 2.35 degrees.
There is only one recovery seed. Do not describe those differences as proven
statistical equivalence or a robust all-axis win without replication.

Cold and warm campaigns both shift thresholds downward, but the fractional
change is not uniformly 35–40% for all three thresholds: the highest threshold
moves from about 47.5 to 39–40 mV. Prefer the actual threshold vectors or
per-threshold percentages. Broader causal claims about readout versus network
effects require controlled comparisons.

The anti-collapse term is a **prediction–truth Pearson-correlation guard**, not
the reported σβ = 7.27°. The empirical residual spread also appears as a detached
normalization in the bias constraint, which is another distinct use of σ.

## 4. The exact seed 40442 result Lindsey asked about

Checkpoint:
`/work/users/das214/SmartPixels/smart-pixels-ml/runs/o24_scratch_bias/s40442/seed_40442/best.weights.hdf5`.

This is the **best composite validation-loss checkpoint, epoch 7,576**, from a
completed 10,000-epoch run. It is neither the final checkpoint nor the historical
minimum plain NLL. Readout is `(8,20)`; thresholds are
`[4.968767166, 16.386762619, 39.236671448]` mV. Full-test plain NLL was reproduced
exactly as **−37,512.9765625** using the existing eight-batch reporting convention.
The cached batch sizes are `[5000,5000,5000,5000,5000,5000,3960,3959]`;
the NLL divisor is eight, not `37919/5000`. Verified checkpoint SHA256:
`6c715a892d6c9c308350bf08e385af0e18c271f95b198e13d99155c829575b73`.

| Target | Global mean residual | Global std | Central 68% half-width |
|---|---:|---:|---:|
| x | +0.03795 µm | 9.69750 µm | 6.75 µm |
| y | −0.07196 µm | 3.57103 µm | 1.68 µm |
| α | −0.14073° | 9.43673° | 1.48° |
| β | +0.12601° | 7.26704° | 1.78833° |

For β, `q16 = −1.711380615°`, `q84 = +1.865270996°`.
Residuals extend from approximately −127.868° to +127.285°.
There are **689 events beyond ±15° (1.8170%)**, contributing **87.3296%** of
`sum((residual - global_mean)**2)`. Beyond ±10° there are 1,133 events (2.9879%),
contributing 90.6156%. State the threshold when quoting a tail percentage.

The response to Lindsey is:

> The 7.27° value is calculated directly as the standard deviation of all
> test-set residuals, including tails; no Gaussian fit was used. The unbinned
> central 68% containment interval is [−1.71°, +1.87°], giving a half-width of
> about 1.8°.

Authoritative new output directories under
`/work/users/das214/SmartPixels/smart-pixels-ml/runs/perf_plots_o24_slides_20260909/`:

- `residual_40442/figures/`: `predictions_cold_40442.npz`, `metrics.json`,
  `provenance.json`, `residuals_cold_40442.{png,pdf}` and
  `beta_residual_cold_40442.{png,pdf}`.
- `residual_linear_40442/figures/`: `metrics.json`, `provenance.json`,
  `residuals_linear_cold_40442.{png,pdf}` and
  `beta_residual_linear_cold_40442.{png,pdf}`. Reuses the preceding cache.

Cache hashes are checked against the checkpoint. The first script forces CPU
inference if the cache is absent. The linear script does no inference.
See `scripts/distillation/make_residual_plots_o24_40442.py:119` and
`scripts/distillation/make_residual_plots_o24_40442_linear.py:76` for the
unbinned calculations; `:65` and `:96` in the latter implement complete-range
histogram bins and linear event counts.

Notebook source: `legacy/notebooks/physics/from_weights.ipynb`, zero-based cells
18/22/26/30. It uses filled `plt.hist`, 50 edges/49 bins, linear counts, no fit.
Its old hardcoded scales `[75,18.75,8,0.5]` and some unit labels must not be
copied into the current dataset. Current metadata scales are
`[123.730140686,30.942258835,6.587896824,1.848129153]`.
Our adaptation converts angles to degrees, extends bins across every residual,
and then zooms the axes only. Linear central windows: x ±40 µm, y ±12 µm,
α/β ±15°. In-view/outside counts respectively:
37,793/126; 37,741/178; 35,819/2,100; 37,230/689.
Both β linear panels share identical bins and counts.

## 5. Loss formulas added to the slides

Before: batch-summed four-dimensional Gaussian NLL plus the inherited
correlation penalty. With scaled truth t, predicted mean μ and covariance Σ:

```text
L_NLL = (1/2) sum_e [(t_e-μ_e)^T Σ_e^-1 (t_e-μ_e) + log det Σ_e + 4 log(2π)]
L_before = L_NLL + P_corr
P_corr = 10000 sum_k [γ_k^+ q_k + (1/2) q_k²]
q_k = max(0, 0.5 - Corr_batch(μ_k,t_k))
```

Now, O22 and O24 use by-term bin balancing and 60 per-bin bias constraints:

```text
L_now = sum_e,k w_ek ℓ_ek + P_corr
        + 5000 sum_k,b [clip(λ_kb,0,5) g_kb + (1/2) g_kb²]
g_kb = clip(|b_kb| / s_k - 0.17, 0, 2)
s_k = stop_gradient(std_batch(r_k))
```

There are four targets and 15 soft bins per target. The conditional NLL terms
ℓ sum to the original NLL in fixed x,y,cotα,cotβ order. Each gets its own
target's inverse-occupancy weight: normalize, clip to [0.25,4], renormalize.
Bias uses soft-bin means; angular bias is in degrees, whereas the bin-balanced
conditional terms use native scaled labels/cotangents. Diagonal Cholesky
entries are softplus plus a small floor. Numerical stabilizers are omitted
above; the native slides' speaker notes document them.

`val_loss` retains the weighted task loss but excludes MDMM constraint
penalties. For cross-arm comparison use **`val_plain_nll`**, not either training
composite or weighted validation loss. The reported campaign NLL is the full
test batch-sum divided by eight test batches, **not per-event NLL**.

Sources: `scripts/slides/build_o24_loss_slides.py`,
`two_bit_optimization_helpers/conditional_nll.py`, `mdmm.py`, and the campaign
driver. No loss implementation was changed in this presentation work.

## 6. The live slide deck and how to continue editing it

**Correct deck:** `/work/users/das214/atrain-slides/o21v2cold.atdeck`.
**Port:** 8900; user URL after forwarding: `http://localhost:8900`.
Read-only HTTP GET `/deck` succeeded on September 14: **revision 603, 51 slides,
16 GIF assets**. The weekly deck on 50696 is a different deck.

The original lattice-pair evolution animations are retained. This Codex work
added 24 slides in four operations (18 + 2 + 2 + 2). Two additional slides and
some rearrangement appeared in the user's deck between updates; do not claim
every difference from the original 25-slide snapshot was made by Codex.

Current absolute positions and stable IDs:

| Position | Content | Stable ID |
|---:|---|---|
| 12–13 | Loss before/now | `sl_o24_20260909_loss_before`, `sl_o24_20260909_loss_now` |
| 21 | Cold 40442 summary/pulls, footer O24 07/18 | `sl_o24_20260909_cold_40442` |
| 22 | Four full-range residuals | `sl_o24_20260909_residual_40442_all` |
| 23 | β log/full-range and linear central detail | `sl_o24_20260909_residual_40442_beta` |
| 24 | Four linear residuals | `sl_o24_20260909_residual_linear_40442_all` |
| 25 | β full-range and central, both linear | `sl_o24_20260909_residual_linear_40442_beta` |

“07/18” is a retained O24 section footer, not the live deck's absolute numbering.
Always identify slides by fresh IDs, since the user can reorder them.

Read `/work/users/das214/atrain-slides/README-AGENT.md` and
`/work/users/das214/atrain-slides/packages/skill/SKILL.md` before editing.
All live edits in this session went through the server's validated operation
path (`/asset?apply=false` upload, then `/op`), **never by replacing live
`deck.json`**. Keep native editable text, tables and equations, using images for
scientific plots. Preserve current user edits, existing assets and animations.
Read `.deckd-token` privately when needed; never print or put it in a handoff.

Reusable scripts in `scripts/slides/`:

- `build_o24_extension.py`: original section and reusable native layout helpers.
- `build_o24_loss_slides.py`: before/now loss equations and detailed notes.
- `build_o24_residual_slides.py`: log/full-range residual explanation.
- `build_o24_linear_residual_slides.py`: notebook-style linear additions.
- `preview_o24_deck.mts`: schema checks, standalone offline HTML, browser
  screenshots, image/text/equation/overflow diagnostics.
- `apply_o24_extension.mjs`: additive API application; verifies original slides,
  ordering and assets, new content and HTTP asset hashes. Stable IDs prevent
  duplicate insertion. An optional adjacent-slide guard catches moved anchors.

Each output directory contains `deck_before.json`, `new_slides.json`, asset
manifests, `deck_preview.json`, `preview.html`, `screenshots/`,
`preview_validation.json`, `deck_after.json`, `live_verification.json` and a
README. The latest standalone full-deck export is also at:
`/work/users/das214/SmartPixels/smart-pixels-ml/runs/perf_plots_o24_slides_20260909/preview.html`.

Old builder snapshots are historical. For a new edit, fetch a fresh snapshot
into a new output directory and review the new native content. Do not rerun an
old builder expecting it to preserve later user edits by reconstructing them.
The applicator only inserts its planned new slides; it is not an updater for
existing human-edited slides.

The actual history generator is
`scripts/distillation/make_history_plots_o24_slides.py`. Some old generated
speaker notes may mention `make_o24_slide_histories.py`; that is not the script
to run. It plotted all 70,012 rows: seven complete 10,000-epoch histories plus
the 12-epoch recovery, with separately labelled early-time zooms.

## 7. Current campaign ledger and unfinished work

The original handoff is still the detailed campaign source. This table comes
from `runs/perf_plots_o24_seeds/perf_summary_o22.json` and refers to **evaluated
saved checkpoints**, not history extrema:

| Arm | Plain NLL | σx [µm] | σy [µm] | σα [°] | σβ [°] |
|---|---:|---:|---:|---:|---:|
| parent | -40,161.789 | 6.644 | 1.929 | 7.782 | 5.566 |
| o22_ep100 | -39,775.082 | 6.660 | 1.948 | 9.090 | 6.464 |
| recovered | -40,336.570 | 6.573 | 1.924 | 8.789 | 6.188 |
| cold_40342 | -38,058.273 | 8.928 | 3.185 | 9.663 | 7.206 |
| cold_40442 | -37,512.977 | 9.697 | 3.571 | 9.437 | 7.267 |
| cold_40542 | -37,720.109 | 9.199 | 3.365 | 12.609 | 9.460 |
| cold_40642 | -37,907.484 | 9.380 | 3.977 | 9.189 | 6.866 |
| warm_41042 | -38,202.445 | 7.210 | 2.169 | 9.174 | 7.124 |
| warm_41142 | -37,874.414 | 7.851 | 2.715 | 9.040 | 6.740 |
| warm_41242 | -38,456.871 | 7.241 | 2.280 | 8.930 | 6.670 |

Cold saved-best epochs: 40342→9496, 40442→7576, 40542→9802, 40642→8915.
Warm plots use final epoch 10000. Recovery uses plain-NLL best epoch 2 of 12.
Exact paths, readouts, selection rules and bias sources:
`runs/perf_plots_o24_slides_20260909/checkpoint_provenance.json`.

September 14 status check (`bash scripts/status_o24.sh`): all seven main AF
histories have 10,000 epochs and DONE markers; no matching AF discovery
trainer was listed; Gautschi queue was empty. The Depot histories printed by
the same status script are **stale migration copies** and are not evidence
that the completed AF jobs need resuming. The script's “best” values are
historical plain-NLL minima, not necessarily the evaluated best-weight file.
Its final reference line also contains older shorthand: use the table above.

Still unfinished in the inspected canonical locations:

- `runs/o24b_recover/` has only `s41142`; replication on 41042 and 41242 has
  not been done here. Do not confuse those with the completed long warm runs.
- Warm 41042 has full-test σ/pull/NLL results but no authoritative per-bin
  bias JSON in the slide provenance. It was not newly evaluated for bias here.
- The existing pull-fit bin-centre fix remains uncommitted, per user instruction.
- The Claude-hosted historical report URLs were not updated; the live ATrain
  deck and repo artifacts are the presentation work completed here.

## 8. QConv2D: code orientation for the next discussion, not a launch recipe

`two_bit_optimization_helpers/models/conv2d_model_quantized.py:38` defines
`QConv2D_Max`: 3×3 separable convolution (five filters), 1×1 convolution,
3×3 average pooling, flatten, two 16-unit dense layers and 14 outputs.
Convolutions/tanh activations use 4-bit quantization; the dense head and
post-pool activation use 8-bit quantization. Saved summaries report 1,869
parameters. The 14 outputs represent four means and ten Cholesky entries.
Weight/activity regularizers are part of the model.

`QConv2D_Max_SoftQuantizer` at `:101` adds the trainable 2-bit input quantizer.
Neither variant has the pair-lattice router. This is QAT; a chip-area/latency
or bit-accurate deployment claim still requires its own hardware validation.

Important existing-code traps:

1. `QConv2D_Max` ignores its `n_filters` argument when calling `_conv_network`
   (`conv2d_model_quantized.py:59–61`); the soft-quantizer variant has the same
   issue around `:114`. A nominal width sweep can silently keep five filters.
   Factory defaults are in `two_bit_optimization_helpers/train.py:148–155`.
2. `train_qconv2d_max_part2_only.py:30–44` uses an older charge/parquet dataset,
   1,000 epochs, patience 50 and retries. Its inherited thresholds are about
   `[226,601,1457]`, **not the O24 mV thresholds**. It hard-digitizes upstream.
   The docstring says ViT although the selected model is QConv2D_Max.
3. `prepare_tfrecords.py:34–43` defaults a two-slice selection to `[0,19]`.
   Establish the actual cached data/time-axis provenance before comparing it
   with O24 or asserting those are the current desired slices.
4. Shared `train.py:187–195` chooses legacy `custom_loss` for Max models;
   `loss.py:5–39` uses ReLU/floor diagonals and clipped probability. That is
   not the unclipped softplus v2 loss used for current ViT results.
5. `scripts/distillation/train_distill_qconv2d.py:40–89` also defaults to old
   teacher/data/threshold paths. `distill14.py:50–66` adds per-event legacy
   data NLL and Gaussian KL; its manual total omits `student.losses`, so the
   regularizer treatment differs from ordinary Keras training.
6. The distillation driver has EarlyStopping, overwrites its CSV, and lacks
   the current campaign's resume/best-per-epoch saving protocol
   (`train_distill_qconv2d.py:101–116`). Its seed argument is passed to loaders
   without fully seeding model initialization. Do not inherit these settings
   silently under the present full-budget/seed-ledger agreement.
7. Do not simply set `SMARTPIX_MODEL_NAME=QConv2D_Max` in the O24 discovery
   driver. It builds all-101-slice input and unconditionally accesses
   `simple_router_output` (`run_simplerouter_mdmm_discovery.py:391,652,919,965`).

Historical QConv runs exist: `runs/qconv2d_max_run_full/summary.json`
(333 epochs, best legacy val loss ≈−16142),
`runs/distill_qconv2d_max/summary.json` (438 epochs, best data loss ≈−3.18348),
`runs/distill_qconv2d_hint/summary.json` (358 epochs, ≈−3.27587),
`runs/distill_qconv2d_takd_hint/summary.json` (257 epochs, ≈−2.92809), and
`runs/qconv2d_confweight/summary.json` (205 epochs, ≈−1.52489).
These are historical, differently normalized/differently configured results,
not numbers to place beside O24 NLL. Inspect the TAKD teacher provenance before
quoting it: model type and checkpoint-path naming do not straightforwardly agree.

`docs/agent-memory/mdmm-harshul-audit.md:48` records earlier collaborator reports
of stuck QConv seeds and an initialization hypothesis. Treat that as historical
evidence to investigate, not a confirmed diagnosis of the proposed new campaign.

Start the discussion with a few questions, then follow up rather than dumping
an implementation questionnaire on the user:

- Is the primary goal a controlled comparison with the current ViT, or the
  best model within a specific chip resource/precision budget?
- Should the first baseline use the same full O24 dataset and fixed
  parent/recovered readout to isolate network capacity, or should thresholds
  and/or slice selection also learn? These are different experiments.
- Keep the existing 1,869-parameter 4-bit-conv/8-bit-head architecture and full
  covariance, or explicitly allow changes to size, precision or outputs?
- Train on labels with v2 NLL, carry over the bias-aware objective, or distill
  from a specified ViT checkpoint? Decide how to isolate those effects.
- Which seeds, complete epoch budgets, checkpoint metric and evaluation
  metrics are approved? Report tails and central containment as well as global
  σ, bias, pulls and comparable likelihood; agree the success criterion first.

Those are open questions. A fixed-readout baseline is a discussion suggestion,
not an agreed decision. No new experiment was named or launched here.

## 9. Commands and environment

Use the established environment; do not replace packages casually:
`/work/users/das214/envs/smartpix-2bit/bin/python` (TF 2.15-era environment;
original handoff records NumPy 1.23.5 and SciPy 1.13.1).

Current O24 TFRecords:
`/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/TFR_files_all101_noise_contained_discovery`.
Full train/test sizes: 155,000 / 37,919. Noise is already in these TFRecords;
the evaluator loads with `noise=-1`. Reports go in the repo, not that data path.

Safe status and source orientation:

```bash
cd /work/users/das214/SmartPixels/smart-pixels-ml
bash scripts/status_o24.sh
git status --short
rg -n 'QConv2D_Max|QConv2D_Max_SoftQuantizer' two_bit_optimization_helpers
cat runs/qconv2d_max_run_full/summary.json
cat runs/distill_qconv2d_max/summary.json
curl --max-time 5 --fail --silent --show-error --output /tmp/smartpix-live-deck.json http://127.0.0.1:8900/deck
```

Reproduce the residual figures using the complete prediction cache:

```bash
cd /work/users/das214/SmartPixels/smart-pixels-ml
/work/users/das214/envs/smartpix-2bit/bin/python scripts/distillation/make_residual_plots_o24_40442.py
/work/users/das214/envs/smartpix-2bit/bin/python scripts/distillation/make_residual_plots_o24_40442_linear.py
```

Native preview example (existing prepared linear-slide output):

```bash
cd /work/users/das214/SmartPixels/smart-pixels-ml
LD_LIBRARY_PATH=/work/users/das214/.pixi-cache/pkgs/nss-3.118-h445c969_0/lib:/work/users/das214/.pixi-cache/pkgs/nspr-4.40-h29cc59b_0/lib:/work/users/das214/.pixi-cache/pkgs/libgbm-1.0.8-h54a6638_0/lib \
node --import /work/users/das214/atrain-slides/node_modules/tsx/dist/loader.mjs \
  scripts/slides/preview_o24_deck.mts runs/perf_plots_o24_slides_20260909/residual_linear_40442
```

Existing-plan verification/dry run (no edits without `--apply`):

```bash
node scripts/slides/apply_o24_extension.mjs runs/perf_plots_o24_slides_20260909/residual_linear_40442
```

For a newly reviewed addition, the same command with `--apply` performs the
authorized operation batch and preservation checks. Do not restart or create
another slide server if 8900 is already serving the correct deck.

## 10. Working-tree state, agreements and restart traps

Git HEAD inspected: `8169442` (“Handoff: record the two env packages that had to
be installed”). No commits were made in this Codex session.

Before writing this handoff, existing modifications were `AGENTS.md` and
`scripts/distillation/make_perf_plots_o22.py`. The latter includes output/arm
environment overrides and the previously uncommitted pull-fit bin-centre fix
around line 97. Preserve it; **do not commit merely because the old handoff
says “Commit this.”** The user's instruction was to leave it uncommitted.

Other pre-existing untracked items: `AGENTS.md.bak-20260908`, the original
`HANDOFF_TO_CODEX_20260907.md`, the three new history/residual plotting scripts,
and `scripts/slides/`. This migration additionally writes this file, the Claude
prompt, the dated handoff pointer and verification snapshots, and updates both
agent routers. Some files/artifacts are ignored by Git: a clone or `git pull`
alone does not transfer the local slide folder, runs, caches or these uncommitted
changes. Continuing Claude in this same workspace avoids that issue.

Agreements that carry forward:

- Run the user's chosen epoch budgets in full; do not silently inherit early
  stopping or offer repeated mid-run stop opinions. Late transitions matter.
- Never subset the dataset without explicit permission. A clearly labelled
  plot-axis zoom is not a training/evaluation sample cut.
- Discuss changes to an experiment's meaning before making them. Routine
  operations such as fixing or restarting a demonstrably broken run can proceed
  within scope. Kill exact PIDs, never broad process-name patterns.
- Push only to the user's `myfork` without further per-push approval; `origin`
  is a collaborator's remote. No commit/push is requested in this handoff task.
- Keep reports and figures in this repo and provide absolute figure paths.
- For O24 reproduction, explicitly set the correct ViT architecture. Do not
  carry that instruction blindly into a new QConv architecture experiment.
- Never apply the cold router-smoothing schedule to a warm-start checkpoint.
  Preserve the inherited pair-sampling distribution; a one-hot pin previously
  removed important regularization. Original handoff sections 4–5 explain why.
- Shared AF hardware: inspect actual job state before launching. Host loadavg
  and `free` do not describe the pod's allocation. Follow current platform rules,
  cgroup accounting and the known shared-GPU agreements; do not infer availability
  from the absence of this project's trainers.
- Launch long work with `setsid nohup …`, monitor logs/checkpoints and completion
  markers. A background wrapper PID is not a reliable completion signal, and
  `pipefail` plus `ps | grep -q` previously broke waiting loops.
- The user's September 10 jet/JEC question was explicitly sent to the wrong
  session and withdrawn. It is unrelated to SmartPixels; do not pursue it here.

Read-only September 14 evidence is in `docs/handoff/verification_20260914/`:
`status_o24.txt`, `worktree_before_handoff.txt`, and `deck_snapshot.json`.
The earlier per-update rendering and live-mutation checks are in the artifact
directories described above. Treat later live state as authoritative if it changes.
