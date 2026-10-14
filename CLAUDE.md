# smart-pixels-ml — agent router

You are in the SmartPixels ML repo (CMS smart-pixel ASIC: 2 of 101 time slices,
2-bit readout, ViT regression of track parameters). Route yourself:

1. **State of the project** → `docs/handoff/HANDOFF_2026-09-14.md`, then
   `HANDOFF_TO_CLAUDE_20260914.md`. Read these before touching runs, drivers,
   results or slides. Keep `HANDOFF_TO_CODEX_20260907.md` for the original
   O21–O24 history; the newer handoff updates its interpretations and priorities.
2. **Running the campaign driver** → `scripts/distillation/run_simplerouter_mdmm_discovery.py`.
   For the O21/O24 ViT recipes, set `SMARTPIX_MODEL_NAME=ViT_MaxDeep_PairLattice`
   (unset = wrong architecture, silently). One `--out` dir per seed; `--extend`
   resumes. This router-specific driver cannot simply be switched to QConv2D.
3. **Evaluating checkpoints** → `scripts/distillation/make_bias_comparison.py`
   (`SMARTPIX_ARMS`, `SMARTPIX_OUT`) and `make_perf_plots_o22.py` (pulls).
   Compare NLLs only as `val_plain_nll`; the training loss is a weighted composite.
4. **Gautschi batch** → `scripts/gautschi/submit_o24_scratch_bias.sbatch`
   (chunk chain, 42 GB cap, resubmit guard). Depot is mounted on both machines.
5. **Figures** → `runs/perf_plots_*`; scripts `scripts/distillation/make_*.py`.
   Print absolute paths for every figure.
6. **Live status without any session state** → `bash scripts/status_o24.sh`
   (both machines, all seeds). The historical follow-ons in
   `scripts/o24_followon_watch.sh` completed in September 7's campaign.
   All seven main O24 histories were DONE at 10,000 epochs on September 14.
   Recheck live status; stale Depot migration copies do not require resuming.
7. **Durable facts from earlier agent sessions** → `docs/agent-memory/`
   (one fact per file; `README.md` is the index). The Claude "Bias and
   Ceiling" artifact URL in the handoff is read-only for non-Claude agents;
   the underlying figures are in `runs/perf_plots_*`.
8. **Platform** → the Purdue AF pod: `/proc/loadavg` and `free` show the host;
   use cgroup files. A100 fits 3–4 trainers. Launch long jobs with
   `setsid nohup … &` and wait on the log, not the pid.

Rules from the user: run full epoch budgets (no early-stop opinions mid-run);
never subset a dataset; ask before changing an experiment's meaning, act on
ops (restart/resubmit/kill broken runs) without asking; push to `myfork`
freely, never to `origin`; keep reports in this repo.

## Start here (2026-09-14, Codex back to Claude)

Read `HANDOFF_TO_CLAUDE_20260914.md` for the session update and
`PROMPT_FOR_CLAUDE_20260914.md` for the requested continuation prompt.
Continue the original Claude context using `HANDOFF_TO_CODEX_20260907.md` where
needed, but reconcile it with the newer handoff's scientific corrections.

The user now wants a shared-understanding discussion about training QConv2D.
No new QConv protocol, seed ledger or epoch budget has been agreed. Start with
the focused questions in the handoff, not a training command. The migration
request authorized documentation and a prompt only; no training was launched.

The correct lattice-animation deck is on port 8900, verified September 14 at
51 slides. Preserve native live edits and use the validated operation API.
Preserve uncommitted work, especially the pull-fit bin-centre correction.
