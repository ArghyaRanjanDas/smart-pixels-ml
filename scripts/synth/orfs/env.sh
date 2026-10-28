#!/usr/bin/env bash
# Shared path/tool checks only; activate your chosen tool environment before use.
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
ORFS_HOME=${ORFS_HOME:-$HERE/ORFS}
WORK_ROOT=${WORK_ROOT:-$HERE/work}
COMMON_DIR=${COMMON_DIR:-$HERE/../hls4ml_common}
ORFS_HOME=$(realpath -m "$ORFS_HOME")
WORK_ROOT=$(realpath -m "$WORK_ROOT")
COMMON_DIR=$(realpath -m "$COMMON_DIR")
PYTHON=${PYTHON:-python3}
require_tools() {
    local tool
    for tool in "$@"; do
        command -v "$tool" >/dev/null || { echo "Missing required tool: $tool. Activate your tool environment or update PATH." >&2; return 2; }
    done
}
require_tag() { [[ ${1:-} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo 'Expected a simple design tag/top identifier' >&2; return 2; }; }
require_rtl_test() {
    : "${RTL_TEST_DIR:?Set RTL_TEST_DIR to the verified RTL test directory}"
    RTL_TEST_DIR=$(realpath "$RTL_TEST_DIR")
    TB=$RTL_TEST_DIR/tb_${TOP}.cpp
    [[ -s $TB && -s $RTL_TEST_DIR/rtl_seq.txt && -s $RTL_TEST_DIR/rtl_stream.txt ]] || { echo "Missing testbench/reference files in $RTL_TEST_DIR" >&2; return 2; }
}

mark_route_complete() {
    "$PYTHON" - "$WORK" "$TOP" "$CLK_NS" "$CLK_PORT" "$UTIL" <<'PYROUTE'
from pathlib import Path
import hashlib,json,sys,os,datetime
work,top,clock,port,util=sys.argv[1:];work=Path(work)
summary=work/'summary.txt';net=work/'results/nangate45'/top/'base/6_final.v'
def sha(p):
    assert p.is_file() and p.stat().st_size>0,'Missing completed route artifact: '+str(p)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
proof=dict(ok=True,top=top,clock_ns=float(clock),clock_port=port,utilization_percent=float(util),
    summary_sha256=sha(summary),netlist_sha256=sha(net),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
p=work/'route_complete.json';temporary=p.with_suffix('.json.tmp')
temporary.write_text(json.dumps(proof,indent=2)+'\n');os.replace(temporary,p)
PYROUTE
}
