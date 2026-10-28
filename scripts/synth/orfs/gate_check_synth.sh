#!/usr/bin/env bash
# gate_check_synth.sh <tag> [netlist-name=1_synth.v]
# RTL_TEST_DIR must contain tb_<top>.cpp and both verified RTL reference files.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
ID=${1:?design tag required}; require_tag "$ID"
WORK=$WORK_ROOT/$ID
TOP=$(sed -n 's/^export DESIGN_NAME *= *//p' "$WORK/config.mk"); require_tag "$TOP"
require_rtl_test
require_tools "$PYTHON" verilator
NET=$WORK/results/nangate45/$TOP/base/${2:-1_synth.v}
[[ -s $NET ]] || { echo "Missing synthesized netlist: $NET" >&2; exit 2; }
CELLS=${CELL_MODELS:-$HERE/nangate45_cells.v}
if [[ ! -s $CELLS ]]; then
    "$PYTHON" "$HERE/lib2v.py" "$ORFS_HOME/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib" > "$CELLS.tmp"
    mv "$CELLS.tmp" "$CELLS"
fi
OUT=$WORK/gate; mkdir -p "$OUT"; cd "$OUT"
rm -f gate_check.txt gate_check_synth.json
cp "$TB" tb.cpp
ulimit -v $(( ${MAX_MEM_GB:-32} * 1024 * 1024 ))
unset CXXFLAGS CPPFLAGS
time verilator --cc --exe --build -j "${NUM_CORES:-8}" -Wno-fatal -Wno-lint -Wno-style -Wno-UNOPTFLAT \
    -MAKEFLAGS 'OPT_SLOW=-O0 OPT_FAST=-O1 OPT_GLOBAL=-O1' --top-module "$TOP" --Mdir obj_dir \
    "$CELLS" "$NET" tb.cpp -o simv > build.log 2>&1
./obj_dir/simv > sim.log 2>&1
for f in rtl_seq.txt rtl_stream.txt; do
    if [[ ! -s $f || $(wc -l < "$f") -ne ${N_CHECK:-200} || $(wc -l < "$RTL_TEST_DIR/$f") -ne ${N_CHECK:-200} ]] || ! cmp -s "$f" "$RTL_TEST_DIR/$f"; then
        echo "FAIL $f: missing, incomplete, or mismatched gate outputs" | tee gate_check.txt
        exit 1
    fi
done
printf 'PASS %s sequential and streamed events\n' "${N_CHECK:-200}" | tee gate_check.txt
"$PYTHON" - "$OUT/gate_check_synth.json" "${N_CHECK:-200}" "$NET" "$RTL_TEST_DIR" <<'PYPROOF'
import json,sys,os,hashlib
from pathlib import Path
path,n,net,ref=sys.argv[1:];n=int(n)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
with open(path+'.tmp','w') as f:json.dump(dict(ok=True,events=n,seq_exact=n,stream_exact=n,netlist_sha256=sha(net),reference_sha256={v:sha(Path(ref)/v) for v in ('rtl_seq.txt','rtl_stream.txt')}),f,indent=2)
os.replace(path+'.tmp',path)
PYPROOF
