#!/usr/bin/env python3
"""Turn a verilator_check.py testbench (tb_<top>.cpp) into a power-activity testbench.

The generated tb keeps reset + the cfg_we/cfg_din weight load exactly as in the RTL
check, then opens a VCD and runs ONLY the stream pass (ap_start held high, one real
validation event per ap_ready, i.e. one new event every clock at II=1), and closes the
VCD when the last output has come out. So the VCD window = steady inference on real
data with the weights already loaded and static -- the operating point we quote power at.

  power_vcd_tb.py <tb_in.cpp> <tb_out.cpp> <vcd_path> [clock_ns=25]
"""
import re, sys
src, dst, vcd = sys.argv[1:4]
clk_ns = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
half_ps = int(round(clk_ns * 1000 / 2))
t = open(src).read()

t = t.replace('#include "verilated.h"',
              '#include "verilated.h"\n#include "verilated_vcd_c.h"\n'
              'static VerilatedVcdC* tfp = nullptr; static uint64_t ts = 0;', 1)
old_tick = re.search(r'static void tick\(\) \{[^\n]*\}', t).group(0)
new_tick = ('static void tick() { top->ap_clk = 1; top->eval(); if (tfp) tfp->dump(ts); ts += %d; cyc++; '
            'top->ap_clk = 0; top->eval(); if (tfp) tfp->dump(ts); ts += %d; }' % (half_ps, half_ps))
t = t.replace(old_tick, new_tick, 1)
t = t.replace('Verilated::commandArgs(argc, argv);',
              'Verilated::commandArgs(argc, argv); Verilated::traceEverOn(true);', 1)
# drop the seq pass
i = t.index('  // ---------- seq pass'); j = t.index('  // ---------- stream pass')
t = t[:i] + t[j:]
# open the VCD at the start of the stream pass, close it right after
k = t.index('  // ---------- stream pass')
t = t[:k] + ('  tfp = new VerilatedVcdC; top->trace(tfp, 99); tfp->open("%s");\n'
             '  fprintf(stderr, "VCD on at cycle %%ld\\n", cyc);\n' % vcd) + t[k:]
k = t.index('  dump("rtl_seq.txt"')
t = t[:k] + ('  top->eval(); tfp->dump(ts); tfp->close(); tfp = nullptr;\n'
             '  fprintf(stderr, "VCD off at cycle %ld\\n", cyc);\n') + t[k:]
t = t.replace('dump("rtl_seq.txt", RS); dump("rtl_stream.txt", RT);',
              'dump("pw_stream.txt", RT);', 1)
t = t.replace('fopen("rtl_timing.json", "w")', 'fopen("pw_timing.json", "w")', 1)
open(dst, 'w').write(t)
