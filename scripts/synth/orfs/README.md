# RTL → ASIC estimates with Yosys and OpenROAD

Use this after the [model export guide](../README.md). The platform is
**Nangate45 typical, 1.10 V, 25 °C**, with a default 25 ns clock and 20% clock
period input/output delays. Results are library-based estimates, not measured
silicon or multi-corner signoff. This flow stops after detailed routing,
parasitic extraction and reporting; it does not run GDS generation.

`setup_orfs.sh` fetches the pinned ORFS flow and Nangate45 files, **not the
Yosys/OpenROAD executables**. Install the compatible tools listed in the parent
guide and put them on PATH. Choose new work directories on suitable storage.

```bash
set -euo pipefail
export ORFS_HOME=/absolute/path/OpenROAD-flow-scripts
# Once, if this checkout does not already exist:
bash "$SP_REPO/scripts/synth/orfs/setup_orfs.sh" "$ORFS_HOME"
export WORK_ROOT="$SP_RUN/orfs"
export NUM_CORES=4 MAX_MEM_GB=16 MIN_FREE_GB=1
export N_CHECK=200
export SP_TOP="$SP_TAG"                 # fixed constants
# For programmable builds instead:
# export SP_TOP="${SP_TAG}_cfgwrap"
```

The memory limit is a virtual-memory cap, not a reservation. Larger designs
may require a larger permitted budget. The free-space check is filesystem
availability, not a user quota check; check your site's quotas separately.

## 1. Sanitize and recheck RTL

The sanitizer handles the known Vitis initialization, ROM and FIFO patterns
and refuses unsupported initialization instead of silently removing behavior.
Recheck the sanitized result before synthesis.

```bash
mkdir -p "$SP_RUN/sanitized_$SP_TAG"
"$PYTHON" "$SP_REPO/scripts/synth/orfs/sanitize_vitis_rtl.py" \
  "$SP_RTL" "$SP_RUN/sanitized_$SP_TAG/$SP_TOP.v"

SP_CFG_ARGS=()
if [[ "$SP_TOP" == *_cfgwrap ]]; then
  SP_CFG_ARGS=(--cfg-hex "$SP_RTL/${SP_TAG}_cfg.hex")
fi
"$PYTHON" "$SP_REPO/scripts/synth/hls4ml_common/verilator_check.py" \
  --rtl "$SP_RUN/sanitized_$SP_TAG" --top "$SP_TOP" --prj "$SP_PROJECT" \
  --x "$SP_X" --y "$SP_Y" --n "$N_CHECK" --jobs "$NUM_CORES" \
  --out "$SP_RUN/vl_sanitized_$SP_TAG" "${SP_CFG_ARGS[@]}"
export RTL_TEST_DIR="$SP_RUN/vl_sanitized_$SP_TAG"
```

`RTL_TEST_DIR` must contain `tb_<top>.cpp`, `rtl_seq.txt`, and
`rtl_stream.txt`. The mapped/routed gates use this exact interface and these
references. Configured testbenches embed the supplied image.

## 2. Map, verify, then route

```bash
STOP_AFTER=synth bash "$SP_REPO/scripts/synth/orfs/route_design.sh" \
  "$SP_RUN/sanitized_$SP_TAG/$SP_TOP.v" "$SP_TOP" 25 ap_clk 40 "$SP_TAG"
bash "$SP_REPO/scripts/synth/orfs/gate_check_synth.sh" "$SP_TAG"
bash "$SP_REPO/scripts/synth/orfs/route_resume.sh" \
  "$SP_RUN/sanitized_$SP_TAG/$SP_TOP.v" "$SP_TOP" 25 ap_clk 40 "$SP_TAG"
```

Yosys/ABC map arithmetic into library cells. `ADDER_MAP_FILE` is deliberately
empty: the tested Yosys version's `extract_fa` mapping produced incorrect
outputs on these models. Do not remove that setting without equivalent gate
checks. The flow then runs placement, clock-tree synthesis, routing and RCX.

Use `route_design.sh` only for a fresh tag. Resume the exact existing source
and constraints with `route_resume.sh`; use a new tag for changes. The driver
discards intermediate ODBs to save storage, so arbitrary backward restarts
are not supported. `KEEP_ODB=1` retains final cleanup intermediates but the
stage-by-stage driver still deflates consumed files.

## 3. Activity-based power and final metrics

```bash
bash "$SP_REPO/scripts/synth/orfs/power_vcd.sh" "$SP_TAG"
bash "$SP_REPO/scripts/synth/orfs/power_report.sh" "$SP_TAG" vcd
"$PYTHON" "$SP_REPO/scripts/synth/report_metrics.py" "$WORK_ROOT/$SP_TAG" \
  --rtl-manifest "$SP_RTL/MANIFEST.json" --out "$WORK_ROOT/$SP_TAG/metrics.json"
```

RTL and gate simulation use **Verilator**. `power_vcd.sh` excludes reset and
configuration loading, checks routed-gate outputs, and shifts the activity
timestamps to start at zero. This avoids the previous VCD-origin power error.
Power analysis uses **OpenSTA inside OpenROAD**, with the routed database,
extracted parasitics and that VCD. It rejects `STA-1452`, failed gate checks
and stale evidence. Activity is zero-delay gate simulation: it does not model
delay-dependent glitches. The `flat`/`default` power options are diagnostics
only and are never accepted as comparable VCD power.

| Quantity | Accepted source / units |
|---|---|
| Routed cell, core and die area | `summary.txt`, `6_finish.rpt`, finish metrics; µm², converted to mm² by dividing by 10⁶ |
| Setup/hold slack, setup TNS | Final extracted timing; ns; positive slack means the constraint is met |
| Latency / event interval (II) | `MANIFEST.json` RTL simulation; cycles and cycles × clock in ns |
| Power | `power/power_vcd.rpt` and verified `power_vcd.json`; W converted to mW × 1000 |
| Final machine-readable result | `metrics.json`; missing stages are `null`, never zero |

Cell area sums standard-cell footprints. Core/die area include floorplan
whitespace and should not be compared interchangeably. II=1 at 25 ns means
one accepted event every 25 ns; it is not slack or latency. The clock used
here is a constraint, not a measured maximum frequency.

Keep `config.mk`, `constraint.sdc`, `summary.txt`, route-completion proof,
`results/nangate45/<top>/base/6_final.{v,odb,spef}`, final timing reports,
`gate/gate_check_synth.json`, `power/gate_check.json`, `power/activity.vcd`,
`power/power_vcd.{rpt,json}` and the RTL `MANIFEST.json`. Hashes bind accepted
power to its routed netlist and VCD. Do not report mapped-only cell area as
routed area, or vectorless power as the validated activity result.
