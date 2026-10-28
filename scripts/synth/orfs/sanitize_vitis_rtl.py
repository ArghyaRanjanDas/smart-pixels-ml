#!/usr/bin/env python3
"""Make Vitis HLS 2024.1 Verilog synthesizable by a generic ASIC flow (yosys/ORFS).

usage: sanitize_vitis_rtl.py <rtl_dir> <out.v> [--files f1.v f2.v ...]

Concatenates the design's .v files (MANIFEST.json 'files' order if present) into one
file and removes the only FPGA-specific constructs Vitis emits for io_parallel hls4ml:

  1. `initial begin #0 r = C; ... end` power-on blocks (FPGA GSR init). Every
     register they touch must also be assigned C in an `if (ap_rst ...)` /
     `if (reset ...)` branch, which this script checks; with the testbench / chip
     holding reset for >=1 cycle the behaviour is identical.
     Also accepts the exact FRP output-FIFO co-simulation initializer only when
     its three state values match the explicit synchronous reset and its two
     diagnostic names are unused outside declarations/comments.
  2. `reg x = C;` declaration initializers in the FIFO wrappers (same argument;
     the FIFOs reset empty_n/full_n under `if (reset)`).
  3. `(* ... *)` synthesis attributes (CORE_GENERATION_INFO, fsm_encoding): pure
     metadata, no function.
Everything else (the *_mul_* modules are behavioural `*`, the FIFO ShiftReg is a
plain register array) is already generic RTL and is kept verbatim.
"""
import json, os, re, sys
from pathlib import Path
import hashlib

rtl, out = sys.argv[1], sys.argv[2]
man = os.path.join(rtl, "MANIFEST.json")
files = json.load(open(man))["files"] if os.path.exists(man) else sorted(
    f for f in os.listdir(rtl) if f.endswith(".v"))

# The post-RTL manifest also contains ROM .dat assets, not Verilog source.
files = [f for f in files if f.endswith(".v")]

init_blk = re.compile(r"^initial begin\n((?:#0 \w+ = [^;]+;\n)*)end\n", re.M)
decl_init = re.compile(r"^(\s*reg\s+(?:\[[^\]]+\]\s*)?)(\w+)\s*=\s*([^;]+);", re.M)
# only `(* name = value *)` attributes; never the `@ (*)` sensitivity list
attr = re.compile(r"\(\*\s*[A-Za-z_]\w*\s*=[^*]*?\*\)")
stats = dict(initial_blocks=0, initial_regs=0, decl_inits=0, attrs=0, unchecked=[])

_assign = re.compile(r"\b(\w+)\s*<=\s*([^;]+?)\s*;")
def has_reset_assign(assigned, reg, val):
    # a (reset-branch) nonblocking assignment reg <= val somewhere in the module (loose check)
    return (reg, val.strip()) in assigned

def lower_fifoout_init(source, filename):
    """Only Vitis FRP output-FIFO reset init plus provably unused diagnostics.

    Unlike legacy #0 initializers this generated block contains plusargs. Its
    three functional state assignments must match the explicit synchronous
    ap_rst branch. Diagnostic names must occur only in their declarations after
    removing this block/comments: a live diagnostic or any extra statement fails.
    """
    strip_comments = lambda text: re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    code = strip_comments(source)
    if len(re.findall(r"^\s*module\b", code, re.M)) != 1 or not re.search(r"\bmodule\s+\w+_frp_fifoout\b", code):
        return source
    blocks = re.compile(r"\binitial\s+begin\b(.*?)^\s*end\b", re.M | re.S)
    expected = re.compile(r"\s*fifo_rdPtr\s*=\s*-1\s*;"
        r"\s*fifo_empty\s*=\s*1'b1\s*;\s*fifo_full\s*=\s*1'b0\s*;"
        r"\s*fail\s*=\s*0\s*;\s*if\s*\(\s*!\s*\$value\$plusargs\("
        r'\s*"warning_suppress_threshold=%0d"\s*,\s*WarningSuppressThreshold\s*\)\s*\)'
        r"\s*WarningSuppressThreshold\s*=\s*5\s*;\s*")
    def lower(match):
        if not expected.fullmatch(strip_comments(match.group(1))):
            return match.group(0)  # Remaining initial check rejects unknown forms.
        reset = re.search(r"always\s*@\s*\(\s*posedge\s+ap_clk\s*\)\s*begin\s*:\s*fifo_ptr"
            r"\s*if\s*\(\s*ap_rst\s*\)\s*begin\s*(.*?)\bend\b", code, re.S)
        wanted = {'fifo_rdPtr': '-1', 'fifo_empty': "1'b1", 'fifo_full': "1'b0"}
        if not reset:
            raise ValueError('Missing synchronous FIFO reset branch: ' + filename)
        body = reset.group(1)
        pairs = _assign.findall(body)
        residue = _assign.sub('', body)
        if residue.strip() or len(pairs) != 3 or dict(pairs) != wanted:
            raise ValueError('FIFO initializer/reset mismatch: ' + filename)
        remainder = strip_comments(source[:match.start()] + source[match.end():])
        for name, declaration in [('fail', r'\breg\s+fail\s*;'),
                                  ('WarningSuppressThreshold', r'\binteger\s+WarningSuppressThreshold\s*;')]:
            if not re.search(declaration, remainder) or len(re.findall(r'\b' + name + r'\b', remainder)) != 1:
                raise ValueError('FIFO co-sim diagnostic is used by logic: ' + filename + ':' + name)
        stats['initial_blocks'] += 1
        stats['initial_regs'] += 3
        stats['fifoout_reset_blocks'] = stats.get('fifoout_reset_blocks', 0) + 1
        return ''
    return blocks.sub(lower, source)

def lower_pipeline_valid_init(source, filename):
    """Vitis FRP valid-vector/count reset, including complete per-bit reset."""
    strip_comments = lambda text: re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    code = strip_comments(source)
    module = re.search(r"\bmodule\s+(\w+_frp_pipeline_valid)\b", code)
    if len(re.findall(r"^\s*module\b", code, re.M)) != 1 or not module:
        return source
    blocks = re.compile(r"\binitial\s+begin\b(.*?)^\s*end\b", re.M | re.S)
    expected = re.compile(r"\s*v2\s*=\s*0\s*;\s*fail\s*=\s*0\s*;\s*nvd_reg\s*=\s*0\s*;"
        r"\s*if\s*\(\s*!\s*\$value\$plusargs\("
        r'\s*"warning_suppress_threshold=%0d"\s*,\s*WarningSuppressThreshold\s*\)\s*\)'
        r"\s*WarningSuppressThreshold\s*=\s*5\s*;\s*")
    def lower(match):
        if not expected.fullmatch(strip_comments(match.group(1))):
            return match.group(0)
        # Width and iteration range must agree, so reset covers every bit.
        if not re.search(r"\breg\s*\[\s*PipelineLatency\s*-\s*2\s*:\s*0\s*\]\s*v2\s*;", code):
            raise ValueError('Unsupported valid-vector width: ' + filename)
        vector_reset = re.search(r"\bgenerate\s+for\s*\(\s*i\s*=\s*0\s*;\s*i\s*<\s*PipelineLatency\s*-\s*1\s*;"
            r"\s*i\s*=\s*i\s*\+\s*1\s*\)\s*always\s*@\s*\(\s*posedge\s+ap_clk\s*\)\s*begin"
            r"\s*if\s*\(\s*ap_rst\s*\)\s*v2\s*\[\s*i\s*\]\s*<=\s*0\s*;\s*else\b", code)
        count_reset = re.search(r"\balways\s*@\s*\(\s*posedge\s+ap_clk\s*\)\s*begin"
            r"\s*if\s*\(\s*ap_rst\s*\)\s*nvd_reg\s*<=\s*0\s*;\s*else\b", code)
        if not vector_reset or not count_reset:
            raise ValueError('Missing complete synchronous valid-vector/count reset: ' + filename)
        # Negative/zero vector extents are outside the proven template. All
        # emitted design instantiations must use an explicit integer latency.
        default = re.search(r"\bparameter\s+PipelineLatency\s*=\s*(\d+)\s*;", code)
        if not default or int(default.group(1)) < 2:
            raise ValueError('Unsupported default PipelineLatency: ' + filename)
        for design_file in files:
            design = strip_comments(Path(rtl, design_file).read_text())
            for instance in re.finditer(r"\b" + re.escape(module.group(1)) + r"\s*#\s*\((.*?)\)\s*\w+\s*\(", design, re.S):
                override = re.search(r"\.PipelineLatency\s*\(\s*([^)]*)\)", instance.group(1))
                if override and (not re.fullmatch(r"\d+", override.group(1).strip()) or int(override.group(1)) < 2):
                    raise ValueError('Unsupported PipelineLatency override: ' + design_file)
        remainder = strip_comments(source[:match.start()] + source[match.end():])
        for name, declaration in [('fail', r'\breg\s+fail\s*;'),
                                  ('WarningSuppressThreshold', r'\binteger\s+WarningSuppressThreshold\s*;')]:
            if not re.search(declaration, remainder) or len(re.findall(r'\b' + name + r'\b', remainder)) != 1:
                raise ValueError('Valid-pipeline diagnostic is used by logic: ' + filename + ':' + name)
        stats['initial_blocks'] += 1
        stats['initial_regs'] += 2
        stats['pipeline_valid_reset_blocks'] = stats.get('pipeline_valid_reset_blocks', 0) + 1
        return ''
    return blocks.sub(lower, source)

parts = []
for f in files:
    s = open(os.path.join(rtl, f)).read()
    s = lower_fifoout_init(s, f)
    s = lower_pipeline_valid_init(s, f)
    # Resolve Vitis state aliases before comparing power-on and reset values.
    aliases = dict(re.findall(r"\b(?:localparam|parameter)\s+(\w+)\s*=\s*([^;]+);", s))
    def resolved(v):
        return aliases.get(v.strip(), v.strip()).strip()
    assigned = {(r, resolved(v)) for r, v in _assign.findall(s)}
    # Lower only fully initialized, read-only ROM arrays. Reject unknown data,
    # partial images, procedural writes and every other initial construct.
    rom_blocks = re.compile(r"^initial begin\s*((?:\$readmemh\([^;]+;\s*)+)end", re.M)
    def lower_rom(block):
        body = block.group(1)
        calls = re.findall(r'\$readmemh\("([^"\n]+)",\s*(\w+)\);', body)
        residue = re.sub(r'\$readmemh\("([^"\n]+)",\s*(\w+)\);', '', body)
        if residue.strip() or not calls:
            raise ValueError('Unsupported ROM initializer')
        assignments = []
        for name, reg in calls:
            declaration = re.search(r'\breg\s+\[([^\]]+)\]\s+' + reg + r'\s*\[0:([^\]]+)\]\s*;', s)
            if not declaration:
                raise ValueError('Missing ROM declaration: ' + reg)
            count_expr = declaration.group(2).replace(' ', '')
            count = resolved(count_expr.removesuffix('-1'))
            count = int(count, 0)
            image = Path(rtl) / name
            words = image.read_text().split()
            if len(words) != count or any(not re.fullmatch('[0-9a-fA-F]+', w) for w in words):
                raise ValueError('Incomplete/unsupported ROM image: ' + str(image))
            if re.search(r'\b' + reg + r'\s*\[[^\]]+\]\s*(?:<=|=(?!=))', s):
                raise ValueError('ROM has procedural writes: ' + reg)
            stats.setdefault('rom_images', {})[str(image)] = hashlib.sha256(image.read_bytes()).hexdigest()
            stats['rom_arrays'] = stats.get('rom_arrays', 0) + 1
            assignments.extend(f"assign {reg}[{i}] = 'h{w};" for i, w in enumerate(words))
        return '\n'.join(assignments)
    # Arrays driven by continuous constants must be nets, not register storage.
    rom_names = [r for b in rom_blocks.finditer(s) for _, r in re.findall(r'\$readmemh\("([^"\n]+)",\s*(\w+)\);', b.group(1))]
    s = rom_blocks.sub(lower_rom, s)
    for reg in rom_names:
        s = re.sub(r'\breg(\s+\[[^\]]+\]\s+' + reg + r'\s*\[)', r'wire\1', s)
    for m in init_blk.finditer(s):
        stats["initial_blocks"] += 1
        for line in m.group(1).splitlines():
            r, v = re.match(r"#0 (\w+) = ([^;]+);", line).groups()
            stats["initial_regs"] += 1
            if not has_reset_assign(assigned, r, resolved(v)):
                stats["unchecked"].append(f"{f}:{r}={v}")
    s = init_blk.sub("", s)
    if re.search(r"^\s*initial\b", s, re.M):
        sys.exit(f"unhandled initial block in {f}")
    def _d(m):
        stats["decl_inits"] += 1
        if not has_reset_assign(assigned, m.group(2), resolved(m.group(3))):
            stats["unchecked"].append(f"{f}:{m.group(2)}={m.group(3)}")
        return f"{m.group(1)}{m.group(2)};"
    s = decl_init.sub(_d, s)
    stats["attrs"] += len(attr.findall(s))
    s = attr.sub("", s)
    parts.append(f"// ---- {f} ----\n{s}\n")
if stats["unchecked"]:
    raise SystemExit("Unverified reset initialization: " + repr(stats["unchecked"]))
open(out, "w").write("".join(parts))
print(json.dumps({**stats, "files": len(files), "out": out}, indent=1))
