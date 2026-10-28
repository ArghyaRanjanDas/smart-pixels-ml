#!/usr/bin/env bash
# power_vcd.sh <tag>: routed zero-delay gate activity after reset/config loading.
# RTL_TEST_DIR is required; no personal-directory testbench discovery.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
ID=${1:?design tag required}; require_tag "$ID"; WORK=$WORK_ROOT/$ID
TOP=$(sed -n 's/^export DESIGN_NAME *= *//p' "$WORK/config.mk"); require_tag "$TOP"
require_rtl_test
require_tools "$PYTHON" verilator
RES=$WORK/results/nangate45/$TOP/base
NET=$RES/6_final.v
[[ -s $NET ]] || { echo "Missing routed netlist: $NET" >&2; exit 2; }
CLK=$("$PYTHON" - "$RES/clock_period.txt" <<'PY'
from pathlib import Path
import sys
value=float(Path(sys.argv[1]).read_text().split()[0])/1000
assert value>0
print(value)
PY
)
CELLS=${CELL_MODELS:-$HERE/nangate45_cells.v}
if [[ ! -s $CELLS ]]; then
    "$PYTHON" "$HERE/lib2v.py" "$ORFS_HOME/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib" > "$CELLS.tmp"
    mv "$CELLS.tmp" "$CELLS"
fi
OUT=$WORK/power; mkdir -p "$OUT"; cd "$OUT"
rm -f gate_check.txt gate_check.json
"$PYTHON" "$HERE/power_vcd_tb.py" "$TB" tb_power.cpp "$OUT/activity.vcd" "$CLK"
ulimit -v $(( ${MAX_MEM_GB:-32} * 1024 * 1024 ))
unset CXXFLAGS CPPFLAGS
time verilator --cc --exe --build -j "${NUM_CORES:-8}" --trace --trace-underscore \
    -Wno-fatal -Wno-lint -Wno-style -Wno-UNOPTFLAT -MAKEFLAGS 'OPT_SLOW=-O0 OPT_FAST=-O1 OPT_GLOBAL=-O1' \
    --top-module "$TOP" --Mdir obj_dir "$CELLS" "$NET" tb_power.cpp -o simv > build.log 2>&1
./obj_dir/simv > sim.log 2>&1
if [[ ! -s pw_stream.txt || $(wc -l < pw_stream.txt) -ne ${N_CHECK:-200} || $(wc -l < "$RTL_TEST_DIR/rtl_stream.txt") -ne ${N_CHECK:-200} ]] || ! cmp -s pw_stream.txt "$RTL_TEST_DIR/rtl_stream.txt"; then
    echo 'FAIL routed gate outputs differ from verified RTL or are incomplete' | tee gate_check.txt
    exit 1
fi
"$PYTHON" "$HERE/normalize_vcd.py" activity.vcd
printf 'PASS routed gate == RTL on %s stream events; VCD origin zero\n' "${N_CHECK:-200}" | tee gate_check.txt
"$PYTHON" - "$OUT/gate_check.json" "${N_CHECK:-200}" "$OUT/activity.vcd" "$NET" "$RTL_TEST_DIR/rtl_stream.txt" <<'PYPROOF'
import json,sys,os,hashlib
path,n,vcd,net,ref=sys.argv[1:];n=int(n)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
with open(path+'.tmp','w') as f:json.dump(dict(ok=True,events=n,stream_exact=n,vcd_sha256=sha(vcd),netlist_sha256=sha(net),reference_sha256=sha(ref),t0=0),f,indent=2)
os.replace(path+'.tmp',path)
PYPROOF
