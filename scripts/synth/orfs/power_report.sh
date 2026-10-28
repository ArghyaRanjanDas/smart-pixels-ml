#!/bin/bash
# Correct power for a routed ORFS design (Nangate45, typical corner, 1.1 V).
#
# usage: power_report.sh <workdir_id> [method]      e.g. power_report.sh M_prog
#   method = vcd     (default; requires verified power_vcd.sh activity)
#                    gate-level zero-delay activity on real validation events, II=1 streaming,
#                    weights loaded beforehand (cfg chain static) -- the number to quote.
#          = flat    vectorless: every net 0.1 toggles/cycle (set_power_activity -global 0.1),
#                    clock nets from the 25 ns clock. Cheap cross-check, no stimulus needed.
#          = default OpenROAD's built-in report (input activity 0.1 PROPAGATED through the logic).
#                    BROKEN for arithmetic datapaths: the transition-density propagation blows up
#                    through multiplier/adder trees and registers (M_prog: 1.01 kW). Shown only
#                    for reference; never quote it.
# Reads 6_final.odb + 6_1_fill.sdc (25 ns clock, propagated) + 6_final.spef (OpenRCX).
# Prints total mW and the internal/switching/leakage and sequential/combinational/clock splits.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
require_tools "$PYTHON" openroad
ID=${1:?design tag required}; require_tag "$ID"
WORK=$WORK_ROOT/$ID
TOP=$(sed -n 's/^export DESIGN_NAME *= *//p' "$WORK/config.mk")
RES=$WORK/results/nangate45/$TOP/base
VCD=$WORK/power/activity.vcd
METHOD=${2:-vcd}
[[ $METHOD == vcd || $METHOD == flat || $METHOD == default ]] || { echo 'Expected vcd, flat or default method' >&2; exit 2; }
rm -f "$WORK/power/power_${METHOD}.json"
if [[ $METHOD == vcd ]]; then
    grep -q '^PASS routed gate == RTL' "$WORK/power/gate_check.txt" || { echo 'No successful routed gate check' >&2; exit 2; }
    "$PYTHON" - "$VCD" "$WORK/power/gate_check.json" "$RES/6_final.v" <<'PYVCD'
import sys,json,hashlib
proof=json.load(open(sys.argv[2]));assert proof['ok'] and proof['events']==proof['stream_exact'] and proof['events']>0 and proof['t0']==0
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==proof['vcd_sha256'],'Activity changed since gate verification'
h=hashlib.sha256()
with open(sys.argv[3],'rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==proof['netlist_sha256'],'Netlist changed since gate verification'
with open(sys.argv[1]) as f:
    times=(int(line[1:]) for line in f if line.startswith('#'))
    first=next(times,None)
    assert first==0,'VCD must start at zero'
    assert any(t>0 for t in times),'VCD needs nonzero duration'
PYVCD
fi
[ -s "$RES/6_final.odb" ] || { echo "no 6_final.odb in $RES (route not finished?)" >&2; exit 1; }
PLAT=$ORFS_HOME/flow/platforms/nangate45
mkdir -p "$WORK/power"
TCL=$(mktemp "$WORK/power/pr_XXXX.tcl")
cat > "$TCL" <<TCLEOF
read_lef {$PLAT/lef/NangateOpenCellLibrary.tech.lef}
read_lef {$PLAT/lef/NangateOpenCellLibrary.macro.mod.lef}
read_liberty {$PLAT/lib/NangateOpenCellLibrary_typical.lib}
read_db {$RES/6_final.odb}
read_sdc {$RES/6_1_fill.sdc}
set_propagated_clock [all_clocks]
read_spef {$RES/6_final.spef}
puts "PW_CLOCK_NS [get_property [lindex [all_clocks] 0] period]"
switch $METHOD {
  vcd     { read_power_activities -scope TOP/$TOP -vcd {$VCD} }
  flat    { set_power_activity -global -activity 0.1 -duty 0.5 }
  default { }
}
report_power -digits 6
TCLEOF
RPT=$WORK/power/power_${METHOD}.rpt
ulimit -v $(( ${MAX_MEM_GB:-24} * 1024 * 1024 ))
openroad -no_init -exit "$TCL" 2>&1 | grep -v "^\[INFO ODB" > "$RPT"
rm -f "$TCL"
if grep -q 'STA-1452' "$RPT"; then echo 'Rejected VCD clock-period warning STA-1452' >&2; exit 1; fi
"$PYTHON" - "$RPT" "$ID" "$TOP" "$METHOD" <<'PYEOF'
import re, sys, json, pathlib, hashlib
rpt, wid, top, method = sys.argv[1:]
t = open(rpt).read()
row = lambda n: [float(x) * 1e3 for x in re.search(rf"^{n}\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", t, re.M).groups()]
tot = row("Total"); clk = re.search(r"PW_CLOCK_NS (\S+)", t).group(1)
ann = re.search(r"Annotated (\d+) pin activities", t)
print(f"design {wid} ({top})  method={method}  clock={float(clk):g} ns" + (f"  vcd-annotated pins={ann.group(1)}" if ann else ""))
print(f"total_mW        {tot[3]:.3f}")
print(f"  internal_mW   {tot[0]:.3f}\n  switching_mW  {tot[1]:.3f}\n  leakage_mW    {tot[2]:.3f}")
for g in ("Sequential", "Combinational", "Clock"):
    r = row(g); print(f"  {g.lower():13s} {r[3]:.3f}  (int {r[0]:.3f} sw {r[1]:.3f} leak {r[2]:.3f})")
print(f"report: {rpt}")
result=dict(design=wid,top=top,method=method,valid_for_comparison=method=='vcd',
    clock_ns=float(clk),total_mW=tot[3],internal_mW=tot[0],switching_mW=tot[1],leakage_mW=tot[2],
    source_report=rpt,scope='routed Nangate45 typical corner, zero-delay gate VCD' if method=='vcd' else 'diagnostic vectorless only')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
result['power_report_sha256']=sha(rpt)
if method=='vcd':
    proof=json.loads(pathlib.Path(rpt).with_name('gate_check.json').read_text())
    result.update(vcd_sha256=proof['vcd_sha256'],netlist_sha256=proof['netlist_sha256'])
pathlib.Path(rpt).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
PYEOF
