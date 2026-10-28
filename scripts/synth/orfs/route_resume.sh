#!/bin/bash
# Run OpenROAD-flow-scripts (Nangate45) on one Verilog design, RTL -> detailed route
# -> final report (RCX/SPEF-based STA + power). No GDS / klayout step.
#
# usage: route_design.sh <verilog> <top> [clock_ns=25] [clock_port=ap_clk] [core_util=40] [tag=<top>]
#
# env:  ORFS_HOME   ORFS checkout (default: <script dir>/ORFS; needs flow/ + flow/platforms/nangate45)
#       WORK_ROOT   where per-design work dirs go (default: <script dir>/work)
#       NUM_CORES   threads for OpenROAD (default 4)
#       KEEP_ODB=1  keep the intermediate .odb files (default: delete all but 6_final.odb)
#       STOP_AFTER=synth  stop after yosys (size probe)
#       MIN_FREE_GB stop if the work filesystem has less free (default 1)
#       MAX_MEM_GB  virtual-memory cap for yosys/openroad (default 16), so a big design
#                   fails on its own instead of pushing the shared box into the OOM killer
#       Tools must already be on PATH; no personal environment is activated.
#
# Output: $WORK_ROOT/<tag>/summary.txt plus logs/ reports/ results/ under that dir.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,13p' "$0"; exit 1
fi
VERILOG=$(readlink -f "$1"); TOP=$2
CLK_NS=${3:-25}; CLK_PORT=${4:-ap_clk}; UTIL=${5:-40}; TAG=${6:-$TOP}

source "$(dirname "$(readlink -f "$0")")/env.sh"
require_tag "$TOP"; require_tag "$TAG"
require_tools "$PYTHON" make yosys openroad
WORK=$WORK_ROOT/$TAG
FLOW=$ORFS_HOME/flow
[ -f "$FLOW/Makefile" ] || { echo "no ORFS flow at $FLOW" >&2; exit 1; }
[ -f "$VERILOG" ] || { echo "no verilog $VERILOG" >&2; exit 1; }


ulimit -v $(( ${MAX_MEM_GB:-16} * 1024 * 1024 ))
test -s "$WORK/config.mk" && test -s "$WORK/src/$TOP.v" || { echo 'No existing synthesis to resume' >&2; exit 2; }
cmp "$VERILOG" "$WORK/src/$TOP.v"
require_tag "$TOP"
test "$(sed -n 's/^export DESIGN_NAME *= *//p' "$WORK/config.mk")" = "$TOP"
test "$(sed -n 's/^export ABC_CLOCK_PERIOD_IN_PS *= *//p' "$WORK/config.mk")" = "$("$PYTHON" -c "print(int(round($CLK_NS*1000)))")"
test "$(sed -n 's/^export CORE_UTILIZATION *= *//p' "$WORK/config.mk")" = "$UTIL" || { echo 'Resume utilization differs from original' >&2; exit 2; }
test "$(sed -n 's/^set clk_port_name[[:space:]]*//p' "$WORK/constraint.sdc")" = "$CLK_PORT" || { echo 'Resume clock port differs from original' >&2; exit 2; }
# A failed/interrupted continuation must not expose an earlier completed result.
rm -f "$WORK/summary.txt" "$WORK/route.done" "$WORK/route_complete.json"
# resume: keep existing src/config/sdc (and mtimes) so make does not rebuild
MK=(make -C "$FLOW" DESIGN_CONFIG="$WORK/config.mk" WORK_HOME="$WORK"
    YOSYS_CMD="$(command -v yosys)" OPENROAD_EXE="$(command -v openroad)"
    NUM_CORES="${NUM_CORES:-4}")
LOGD=$WORK/logs/nangate45/$TOP/base
REPD=$WORK/reports/nangate45/$TOP/base
RESD=$WORK/results/nangate45/$TOP/base

echo "== ORFS route: $TOP  clk=${CLK_NS}ns port=$CLK_PORT util=$UTIL  work=$WORK"

# Disk-lean driver. ORFS keeps every intermediate .odb (and copies of some); for
# big designs that is several GB. We build the flow one file target at a time and,
# once a file has been consumed, replace it with an EMPTY file carrying the same
# mtime ("deflate"), so make still sees the chain as up to date and never rebuilds.
deflate() {
  for f in "$@"; do
    [ -s "$f" ] || continue
    touch -r "$f" "$f.ts" && rm -f "$f" && mv "$f.ts" "$f"
  done
}
check_disk() {
  "$PYTHON" - "$WORK" "${MIN_FREE_GB:-1}" <<'PYDISK'
import os,sys
v=os.statvfs(sys.argv[1]);available=v.f_bavail*v.f_frsize
if available < float(sys.argv[2])*1024**3:
    raise SystemExit('Insufficient filesystem availability (not a user quota check); stopping')
PYDISK
}
step() { check_disk; "${MK[@]}" "$@"; }

step synth
[ "${STOP_AFTER:-}" = synth ] && { grep -E "Number of cells|Chip area" "$REPD/synth_stat.txt" | tail -2; exit 0; }
step "$RESD/2_floorplan.odb" "$RESD/2_floorplan.sdc"
deflate "$RESD"/2_[1-6]_*.odb
prev=2_floorplan
for t in 3_1_place_gp_skip_io 3_2_place_iop 3_3_place_gp 3_4_place_resized 3_5_place_dp \
         3_place 4_1_cts 4_cts 5_1_grt 5_2_fillcell 5_3_route 5_route 6_1_fill; do
  step "$RESD/$t.odb"
  deflate "$RESD/$prev.odb"
  [ "$prev" = 2_floorplan ] && deflate "$RESD/1_1_yosys.v" "$RESD/1_synth.v"
  prev=$t
done
step route   # remaining .sdc copies
# Final report only (RCX extraction -> SPEF, post-route STA, power, metrics); no GDS/klayout.
step "$LOGD/6_report.log"
deflate "$RESD/6_1_fill.odb"

# ---- summary --------------------------------------------------------------
"$PYTHON" - "$WORK" "$TOP" "$CLK_NS" > "$WORK/summary.txt" <<'EOF'
import json, re, sys, glob, os
work, top, clk = sys.argv[1:]
base = lambda d: os.path.join(work, d, "nangate45", top, "base")
m = {}
for f in sorted(glob.glob(os.path.join(base("logs"), "*.json"))):
    try: m.update(json.load(open(f)))
    except Exception: pass
def g(*keys):
    for k in keys:
        if k in m: return m[k]
    return None
log6 = open(os.path.join(base("logs"), "6_report.log")).read()
rpt6p = os.path.join(base("reports"), "6_finish.rpt")
rpt6 = open(rpt6p).read() if os.path.exists(rpt6p) else ""
drt = open(os.path.join(base("logs"), "5_3_route.log")).read()
viol = re.findall(r"Number of violations = (\d+)", drt)
pw = re.search(r"^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)%", rpt6, re.M)
groups = {}
for name in ("Sequential", "Combinational", "Clock", "Macro", "Pad"):
    mm = re.search(rf"^{name}\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", rpt6, re.M)
    if mm: groups[name] = float(mm.group(4))
area = re.search(r"Design area ([\d.]+) u\^2 ([\d.]+)% utilization", log6)
print(f"design            {top}")
print(f"clock_ns          {clk}")
print(f"cell_area_um2     {g('finish__design__instance__area') or (area and area.group(1))}")
print(f"stdcell_area_um2  {g('finish__design__instance__area__stdcell')}")
print(f"utilization       {g('finish__design__instance__utilization') or (area and area.group(2))}")
print(f"die_area_um2      {g('finish__design__die__area')}")
print(f"core_area_um2     {g('finish__design__core__area')}")
print(f"instance_count    {g('finish__design__instance__count')}")
print(f"wns_ns            {g('finish__timing__setup__ws')}")
print(f"tns_ns            {g('finish__timing__setup__tns')}")
print(f"hold_ws_ns        {g('finish__timing__hold__ws')}")
print(f"fmax_mhz          {g('finish__timing__fmax')}")
if pw:
    print(f"diagnostic_vectorless_power_total_W     {pw.group(4)}  (internal {pw.group(1)}, switching {pw.group(2)}, leakage {pw.group(3)})")
for k, v in groups.items():
    print(f"power_{k.lower():13s} {v}")
print(f"diagnostic_vectorless_power_total_W_m   {g('finish__power__total')}")
print(f"drc_violations    {viol[-1] if viol else g('detailedroute__route__drc_errors')}")
print(f"route_wirelength  {g('detailedroute__route__wirelength')}")
print(f"spef              {'yes' if os.path.exists(os.path.join(base('results'), '6_final.spef')) else 'no'}")
EOF
cat "$WORK/summary.txt"

if [ "${KEEP_ODB:-0}" != 1 ]; then
  find "$RESD" -name '*.odb' ! -name '6_final.odb' -delete
  rm -rf "$RESD"/*.def "$WORK/objects"
fi

mark_route_complete
