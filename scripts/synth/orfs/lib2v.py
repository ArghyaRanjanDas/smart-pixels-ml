#!/usr/bin/env python3
"""Generate zero-delay Verilog functional models for every Nangate45 liberty cell
(combinational via `function`, flops via `ff` groups; latches/ICG via `latch`/`statetable`
are emitted only for the simple CLKGATE cells; others are skipped with a warning).
Used to simulate the ORFS 6_final.v netlist with Verilator for a VCD-based power check."""
import re, sys
txt = open(sys.argv[1]).read()
out = ["// auto-generated from %s by lib2v.py -- zero-delay functional models" % sys.argv[1],
       "/* verilator lint_off UNOPTFLAT */", "/* verilator lint_off MULTIDRIVEN */"]

def blocks(s, kw):
    """yield (name, body) of top-level `kw (name) { ... }` groups inside s"""
    for m in re.finditer(r'\b%s\s*\(([^)]*)\)\s*\{' % kw, s):
        i = m.end(); depth = 1
        while depth:
            c = s[i]; depth += (c == '{') - (c == '}'); i += 1
        yield m.group(1).replace('"','').strip(), s[m.end():i-1], m.start()

def vexpr(f):
    f = f.replace('"', '').strip()
    f = re.sub(r"(\w+)\s*'", r"(!\1)", f)           # postfix '
    f = f.replace('!', '~').replace('+', '|').replace('*', '&')
    # implicit AND by whitespace between operands: "A B" -> "A & B"
    f = re.sub(r'([\w)])\s+(?=[\w(~])', r'\1 & ', f)
    return f

def top_level_cells(s):
    for name, body, _ in blocks(s, 'cell'):
        yield name, body

skipped = []
for cname, body in top_level_cells(txt):
    pins = {}
    for pname, pbody, _ in blocks(body, 'pin'):
        d = re.search(r'direction\s*:\s*"?(\w+)', pbody)
        fn = re.search(r'\bfunction\s*:\s*"([^"]*)"', pbody)
        if d and d.group(1) == 'internal': continue
        pins[pname] = (d.group(1) if d else 'input', fn.group(1) if fn else None)
    if not pins:
        out += ["module %s();" % cname, "endmodule"]  # FILLCELL/TAPCELL: no signal pins
        continue
    ffs = list(blocks(body, 'ff'))
    lat = list(blocks(body, 'latch'))
    st = list(blocks(body, 'statetable'))
    hdr = "module %s(%s);" % (cname, ", ".join(pins))
    lines = [hdr]
    for p, (d, _) in pins.items():
        lines.append("  %s %s;" % ("output" if d == 'output' else "input", p))
    if ffs:
        (vars_, fb, _), = ffs
        iq, iqn = [v.strip().strip('"') for v in vars_.split(',')]
        g = lambda k: (re.search(r'\b%s\s*:\s*"([^"]*)"' % k, fb) or [None, None])[1]
        nxt, clk, clr, pre = g('next_state'), g('clocked_on'), g('clear'), g('preset')
        lines.append("  reg %s;" % iq)
        sens = ["posedge %s" % clk]
        if clr: sens.append(("negedge %s" % clr.strip()[1:]) if clr.strip().startswith('!') else "posedge %s" % clr)
        if pre: sens.append(("negedge %s" % pre.strip()[1:]) if pre.strip().startswith('!') else "posedge %s" % pre)
        lines.append("  always @(%s)" % " or ".join(sens))
        cond = []
        if clr: cond.append("if (%s) %s <= 1'b0;" % (vexpr(clr), iq))
        if pre: cond.append("if (%s) %s <= 1'b1;" % (vexpr(pre), iq))
        cond.append("%s <= %s;" % (iq, vexpr(nxt)))
        lines.append("    " + " else ".join(cond))
        lines.append("  wire %s = ~%s;" % (iqn, iq))
    elif lat:
        (vars_, lb, _), = lat
        iq, iqn = [v.strip().strip('"') for v in vars_.split(',')]
        g = lambda k: (re.search(r'\b%s\s*:\s*"([^"]*)"' % k, lb) or [None, None])[1]
        en, dd = g('enable'), g('data_in')
        lines.append("  reg %s;" % iq)
        lines.append("  always @* if (%s) %s = %s;" % (vexpr(en), iq, vexpr(dd)))
        lines.append("  wire %s = ~%s;" % (iqn, iq))
    elif st:
        # Nangate CLKGATE(TST)_X*: GCK = CK & latched(E|SE) when CK low
        ins = [p for p, (d, _) in pins.items() if d == 'input']
        en = " | ".join(p for p in ins if p in ('E', 'SE'))
        lines.append("  reg IQ;")
        lines.append("  always @* if (!CK) IQ = %s;" % en)
        lines.append("  assign GCK = CK & IQ;")
    for p, (d, fn) in pins.items():
        if d == 'output' and fn:
            lines.append("  assign %s = %s;" % (p, vexpr(fn)))
        elif d == 'output' and not fn and not (ffs or lat or st):
            lines.append("  assign %s = 1'b0;" % p)  # e.g. LOGIC0
    lines.append("endmodule")
    out += lines
print("\n".join(out))
