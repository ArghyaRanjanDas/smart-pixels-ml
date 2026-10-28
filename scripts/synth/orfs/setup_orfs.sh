#!/bin/bash
# Fetch a minimal OpenROAD-flow-scripts checkout (flow/ + Nangate45 only, ~19 MB)
# pinned to the commit that matches the conda 'asic' env tools:
#   yosys 0.38+92 (2024-02-22), OpenROAD 2.0-12381-g01bba3695 (2024-03-02).
# ORFS 47ed285 (2024-02-29) pins yosys 0.38 and OpenROAD 6011bbe (2024-02-23),
# so its tcl scripts use the same command API as the installed binaries.
#
# usage: setup_orfs.sh [dest=<script dir>/ORFS]
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
DEST=${1:-${ORFS_HOME:-$HERE/ORFS}}
COMMIT=47ed28588427c75486322c73c5ed159a33399f17

git clone -q --filter=tree:0 --no-checkout \
  https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git "$DEST"
cd "$DEST"
git sparse-checkout init --no-cone
cat > .git/info/sparse-checkout <<'EOF'
/flow/*
!/flow/platforms/*
/flow/platforms/nangate45/
!/flow/designs/*
/flow/designs/nangate45/gcd/
/flow/designs/src/gcd/
!/flow/tutorials/
!/flow/test/
EOF
git checkout -q "$COMMIT"
git rev-parse HEAD
printf "ORFS checkout: %s\n" "$DEST"
