# Symbolic MoE and baseline ASIC synthesis

Start here on `symbolic_distillation`. This directory contains the hardware path
for the **four-expert digi16-nf symbolic MoE**, **Paper MLP (`QMlp_Max`)** and
**Paper QConv2D (`QConv2D_Max`)**. It does not require the other architecture
studies or their training drivers.

```text
checkpoint + full validation cache
  → export digitized inputs and coefficients
  → hls4ml / custom symbolic kernels → fixed-point C++
  → Vitis HLS → Verilog RTL
  → Verilator checks → ASIC-safe RTL
  → Yosys + ABC → mapped cells → gate checks
  → OpenROAD / ORFS → placement, CTS, routing, extraction, timing
  → Verilator routed-gate activity → corrected VCD → OpenSTA power
```

**Yosys/OpenROAD consume RTL; they do not replace the C++-to-RTL compiler.**
This path uses Vitis HLS, not Bambu. Existing Bambu-generated generic RTL can
enter the physical flow, but its interface needs its own matching testbench;
the supplied verification helpers target our hls4ml interface.

## Tools and inputs

Use a Python environment containing TensorFlow, QKeras, NumPy and hls4ml, with
`g++`, `verilator`, `vitis_hls`, `yosys`, `openroad`, `make`, and Git available
in the stages where they are needed. Vitis HLS needs its normal installation
and licensing. The scripts do not activate a personal environment.

Verified environment used for this port:

| Component | Version |
|---|---|
| Python / TensorFlow / Keras | 3.10.22 / 2.15.1 / 2.15.0 |
| QKeras / hls4ml / NumPy | 0.9.0 / 1.1.0 / 1.23.5 |
| Vitis HLS | 2024.1 |
| Verilator | 5.023 devel |
| Yosys | 0.38+92, `84116c9a3` |
| OpenROAD | 2.0-12381-g01bba3695 |
| ORFS | `47ed28588427c75486322c73c5ed159a33399f17` |

The Python export can run on a different machine from the hardware tools.
Transfer the **whole export directory**, not only its `.npz`. Use paths without
spaces. Checkpoints and datasets are not committed to Git.

The cache must contain the **full** validation set: analog `X` of shape
`(N,16,16,2)` and normalized `y` of shape `(N,4)`. The symbolic MoE exporter
accepts either `X`/`y` arrays or existing `n`, `X0`/`y0`, … preslice batches.
The baseline MLP/QConv exporter requires the batched preslice format
(`n`, `X0`/`y0`, …), preserving those original batch boundaries. The selected time slices must be the same
ones used in training (our comparison uses 10 and 21). Target order is
`x, y, cotα, cotβ`. Do not change normalization silently. ADC thresholds and
levels are taken from the loaded checkpoint; constructor overrides are exposed
for compatible checkpoints. Digitization is outside the synthesized core.

Set paths in your activated environment:

```bash
set -euo pipefail
export SP_REPO="$(pwd)"                  # this repository root
export SP_RUN="$SP_REPO/runs/shiqi_hw"
export SP_CACHE=/absolute/path/full_validation_cache.npz
export SP_WEIGHTS=/absolute/path/best.weights.hdf5
export PYTHON="$(command -v python)"      # activated analysis/HLS environment
export CUDA_VISIBLE_DEVICES=''
export TF_USE_LEGACY_KERAS=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
mkdir -p "$SP_RUN"
```

Each conversion uses a **new project/output directory**. Keep failed builds
for diagnosis and use a new tag for a changed model. `predict` performs C++
checks without invoking Vitis. `csynth` performs those checks and then HLS;
choose it directly when RTL is wanted, rather than reusing a `predict` output.

## Main path: symbolic MoE digi16-nf

The included adapter implements the existing four signed-barycenter experts,
36→16→4 router, two-time-code sum≥2 noise-filter support, and shared 32→8→10 uncertainty
network. It returns four means and ten raw covariance-factor parameters.
Softplus/covariance decoding and NLL evaluation are outside the hardware core.

```bash
"$PYTHON" scripts/synth/hls4ml_symmoe/export_digi16nf.py \
  --weights "$SP_WEIGHTS" --validation-cache "$SP_CACHE" --out "$SP_RUN/export_moe"
"$PYTHON" scripts/synth/hls4ml_symmoe/prep_digi16nf_val.py \
  --export "$SP_RUN/export_moe/digi16nf_export.npz" \
  --out "$SP_RUN/export_moe/digi16nf_val.npz"

export SP_TAG=moe_const                  # use moe_prog for programmable
export SP_MODEL_WORK="$SP_RUN/$SP_TAG"
"$PYTHON" scripts/synth/hls4ml_symmoe/convert_digi16nf.py \
  "$SP_TAG" "$SP_RUN/export_moe/digi16nf_val.npz" csynth --work "$SP_MODEL_WORK"
export SP_RTL="$SP_RUN/rtl_$SP_TAG"
"$PYTHON" scripts/synth/hls4ml_symmoe/post_rtl_digi16nf.py \
  --work "$SP_MODEL_WORK" --out "$SP_RTL" --jobs 4 --events 200

export SP_PROJECT="$SP_MODEL_WORK/prj_$SP_TAG"
export SP_X="$SP_MODEL_WORK/tb_in/x.npy"
export SP_Y="$SP_MODEL_WORK/tb_in/y.npy"
```

The `_const` or `_prog` suffix selects the implementation. To build both, run
the conversion/post-RTL commands separately with each tag; reuse the export.

**This is not an arbitrary PySR-expression compiler.** Shiqi can reuse the
physical flow for new models, but a changed symbolic equation, router, feature
layout, output count or coefficient type requires a corresponding adapter.
The relevant files are:

| File | Role |
|---|---|
| `hls4ml_symmoe/symmoe_nf_hls4ml.py` | hls4ml custom operations and C++ templates |
| `hls4ml_symmoe/nnet_symmoe_nf.h` | synthesizable fixed-point arithmetic |
| `hls4ml_symmoe/symmoe_nf_hw.py` | widths, configuration arrays and hardware contract |
| `hls4ml_symmoe/symmoe_nf_ref.py` | NumPy float/fixed reference and NLL comparison |
| `hls4ml_symmoe/test_kernel_gxx.py` | standalone kernel/reference comparison |

For an altered model, update its reference and kernel together and rerun all
boundaries below. Do not route an unrelated formula under the digi16-nf name.

## Paper MLP or QConv2D

Choose `QMlp_Max` or `QConv2D_Max` and the corresponding trained checkpoint.

```bash
export SP_MODEL=QMlp_Max                 # or QConv2D_Max
"$PYTHON" scripts/synth/hls4ml_models/export.py --model "$SP_MODEL" \
  --weights "$SP_WEIGHTS" --validation-cache "$SP_CACHE" --out "$SP_RUN/export_$SP_MODEL"

export SP_TAG=mlp_const                  # e.g. qconv_const, mlp_prog, qconv_prog
export SP_PROJECT="$SP_RUN/prj_$SP_TAG"
export SP_BUNDLE="$SP_RUN/export_$SP_MODEL/bundle.npz"
"$PYTHON" scripts/synth/hls4ml_models/convert.py \
  "$SP_TAG" "$SP_BUNDLE" csynth --out "$SP_PROJECT"
export SP_RTL="$SP_RUN/rtl_$SP_TAG"
"$PYTHON" scripts/synth/hls4ml_models/post_rtl.py \
  "$SP_TAG" "$SP_PROJECT" "$SP_BUNDLE" --out "$SP_RTL" --jobs 4

export SP_X="$SP_PROJECT/post_rtl/x.npy"
export SP_Y="$SP_PROJECT/post_rtl/y.npy"
```

For programmable baselines, use a new `_prog` tag and add `--programmable`
to **both** `convert.py` and `post_rtl.py`. Unlike the MoE converter, the
baseline converter uses the explicit flag, not the tag suffix, to select mode.

## Shared ASIC stage and reporting

Continue with [orfs/README.md](orfs/README.md) using the `SP_*` variables above.
That guide covers sanitizer verification, mapped-gate checks, routing, corrected
VCD power and the final unit-labelled JSON. It is the same physical flow for
all three supported architectures.

| Boundary | Required evidence / scope |
|---|---|
| Checkpoint → export | All supplied validation events; analog/code prediction identity; baseline saved-model reload identity |
| MoE → fixed reference | Full-cache float agreement, separately reported fixed-point error and NLL shift |
| hls4ml C++ | All supplied validation events; exact baseline Keras or MoE integer-reference match |
| Generated C++ testbench | Default 2,000 events, separate executable/weight loading |
| RTL / sanitized RTL | Default 200 sequential **and** 200 streaming events; measured latency and event interval |
| Programmable wrapper | Loaded configuration reproduces predictions; zero image changes predictions; MoE also checks packed integer-array identity |
| Mapped netlist | Same sequential/streamed references before expensive placement |
| Routed netlist / power | 200 streamed events, exact reference match; VCD reset to t=0 after configuration loading |

These limited RTL/gate tests are not full-dataset RTL verification or formal
equivalence. FPGA LUT/DSP/FF figures in `*_csynth.rpt` are **HLS estimates of
the FPGA core**, not ASIC area and not a routed FPGA implementation. The
programmable wrapper's configuration flip-flops are additional hardware.

Fixed coefficients become constants and can be optimized away. Programmable
coefficients are loaded into a shift-register wrapper after fabrication; the
topology, dimensions and arithmetic widths remain fixed. Configuration
loading is excluded from the reported steady-state inference activity.

## Provenance and local checks

Ported from the validated hardware implementations at source revision
`d02d428352f9fe4e40e80a5bf352ba1e555aa227` and the corrected ORFS scripts on
the synthesis host (2026-10-28). Portability changes replace personal paths
with explicit arguments and add fresh-output/provenance guards. Existing
symbolic training/distillation files are retained.

```bash
"$PYTHON" scripts/synth/hls4ml_models/test_export.py
"$PYTHON" scripts/synth/orfs/test_flow.py
"$PYTHON" scripts/synth/orfs/test_sanitize_rom.py
"$PYTHON" scripts/synth/orfs/test_sanitize_fifoout.py
"$PYTHON" scripts/synth/test_report_metrics.py
```

The ORFS flow tests use mocked tools to exercise failure handling; they do not
claim a new physical synthesis. Real-checkpoint export validation and compiled
C++ validation for this port are recorded in [VALIDATION.md](VALIDATION.md).
