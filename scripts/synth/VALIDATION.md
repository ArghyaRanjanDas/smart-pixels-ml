# Port validation — 2026-10-28

This records checks of the portable scripts on `symbolic_distillation`, not a
new architecture experiment or new physical-synthesis result.

| Model | Implementation | Full-cache compiled C++ | Separate C++ executable |
|---|---|---:|---:|
| Paper MLP (`QMlp_Max`) | Fixed constants | 37,919/37,919 exact | 2,000/2,000 exact |
| Paper MLP (`QMlp_Max`) | Programmable parameters | 37,919/37,919 exact | 2,000/2,000 exact |
| Paper QConv2D (`QConv2D_Max`) | Fixed constants | 37,919/37,919 exact | 2,000/2,000 exact |
| Paper QConv2D (`QConv2D_Max`) | Programmable parameters | 37,919/37,919 exact | 2,000/2,000 exact |
| Symbolic MoE digi16-nf | Fixed constants | 37,919/37,919 exact | 2,000/2,000 exact |
| Symbolic MoE digi16-nf | Programmable parameters | 37,919/37,919 exact | 2,000/2,000 exact |

Baseline C++ is compared to the quantized Keras checkpoint. Symbolic MoE C++
is compared to its explicit integer/fixed-point reference; this does not mean
that fixed-point arithmetic exactly equals the floating-point MoE. The latter
quantization change is reported separately by `prep_digi16nf_val.py`.

All three real-checkpoint exports passed analog-versus-digitized prediction
identity on the full cache. Both baseline saved-model reloads passed identity.
The MoE float reference agreed with Keras to maximum absolute error
1.43427×10⁻⁶, below the 2×10⁻⁵ check threshold. The six conversion paths ran
with the verified hls4ml environment listed in the main guide, CPU-only.

[validation_20261028.json](validation_20261028.json) records checkpoint hashes,
bundle hashes, exact counts and tested-source hashes. These source hashes were
checked against the actual copied files after the remote executions. Local
raw logs and proof JSONs are retained under `runs/synth/validation/` in the
port worktree; datasets, model weights and generated projects are excluded
from Git.

Additional checks passed:

- Baseline export tests for both factories, including saved-model reload and
  trained ADC thresholds.
- Five mocked ORFS flow tests: mismatches/incomplete outputs fail, changed
  resume settings fail, failed continuation invalidates completion evidence,
  accepted completion hashes agree, stale/bad VCD power is rejected.
- Four FRP/FIFO sanitizer tests plus ROM lowering, alias-reset and partial-ROM
  rejection fixtures.
- Metric collection: µm²→mm², cycles→ns including equal latency ranges,
  absent-stage handling, failed RTL manifest rejection, vectorless power
  exclusion, stale route and stale power rejection.
- Shell syntax and Python compilation checks for the packaged sources.

No new Vitis synthesis, RTL simulation or Yosys/OpenROAD route was launched
for this packaging task. The downstream scripts are ports of the existing
verified flow, with explicit paths and stronger failure/provenance checks.
The mocked tests validate orchestration and rejection behavior, not physical
implementation. A new checkpoint or architecture must run the actual RTL,
gate and physical stages described in the guide before quoting ASIC metrics.
