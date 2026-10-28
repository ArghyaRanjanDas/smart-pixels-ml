"""Source patches every SmartPixels hls4ml project needs (hls4ml 1.1.0 + Vitis HLS 2024.1). Numerics unchanged.

Order in every conversion script:
    hm.write()
    apply_all(odir, tag, programmable=...)   # patches the written project
    hm._compile()                            # C model built FROM the patched sources (NOT hm.compile(): it rewrites)
    ... predict / build(...)  then cosim_check.check_project(...)

  patch_avgpool        avg-pool signed/unsigned divide (RTL wrong for negative averages otherwise). REQUIRED.
  patch_hard_tanh      hard_tanh PIPELINE pragma hoisted out of an if-block (else the activation is not pipelined).
  unpin_multipliers    programmable designs only: drop  #pragma HLS ALLOCATION operation instances=mul limit=...
                       hls4ml sizes that limit for constant weights: dense/conv subtract the trained zero weights
                       (n_zeros) and the conv kernels do not multiply it by n_pixels (PF). With weights as ports
                       nothing folds, so the limit forces sharing: A_prog II 160 / latency 409, B_lut_prog II 2.
                       RF is 1 everywhere, so removing it is the intended fully-parallel datapath.
  patch_tb_precision   testbench prints results with 17 significant digits (stock: 6, ambiguous above |y|~8 for
                       14 fractional bits); cosim_check still snaps to the LSB grid.
  cosim_no_trace       cosim_design -trace_level none (full waveform dumps of these designs are huge).
  raise_fd_limit       RLIMIT_NOFILE soft -> hard for this process (inherited by vitis_hls). Vitis cosim's C-TB stage
                       opens 2 TV files per RTL port at once; B_lut_prog has 1335 ports (codes and the KAN tables /
                       biases are split per element by the layer's ARRAY_PARTITION) -> 2670 files > the default 1024
                       -> "ERROR: Error on TV file" (fopen failure) and no RTL results at all.
"""
import os, re, resource
from patch_avgpool import patch_avgpool  # noqa: F401  (re-export)

_HT_OLD = "    if (CONFIG_T::io_type == io_parallel) {\n        #pragma HLS PIPELINE\n    }\n"
_HT_NEW = "    #pragma HLS PIPELINE\n"


def patch_hard_tanh(odir):
    fn = os.path.join(odir, 'firmware', 'nnet_utils', 'nnet_activation.h')
    s = open(fn).read()
    if _HT_OLD not in s:
        assert _HT_NEW in s, f'hard_tanh pattern not found in {fn}'
        return False
    open(fn, 'w').write(s.replace(_HT_OLD, _HT_NEW)); print('patched hard_tanh pragma', flush=True)
    return True


_ALLOC = re.compile(r'^[ \t]*#pragma HLS ALLOCATION operation instances=mul limit=[^\n]*\n', re.M)


def unpin_multipliers(odir, files=('nnet_dense_latency.h', 'nnet_conv1d_latency.h', 'nnet_conv2d_latency.h',
                                   'nnet_sepconv2d_latency.h', 'nnet_sepconv1d_latency.h', 'nnet_batchnorm.h')):
    n = 0
    for f in files:
        fn = os.path.join(odir, 'firmware', 'nnet_utils', f)
        if not os.path.exists(fn):
            continue
        s = open(fn).read()
        s2, k = _ALLOC.subn('    // (smartpix) multiplier ALLOCATION limit removed: programmable weights, RF=1\n', s)
        if k:
            open(fn, 'w').write(s2); n += k
    print(f'removed {n} multiplier ALLOCATION pragmas', flush=True)
    return n


def patch_tb_precision(odir, tag):
    fn = os.path.join(odir, f'{tag}_test.cpp')
    s = open(fn).read()
    if 'setprecision(17)' in s:
        return False
    a = '    std::ofstream fout(RESULTS_LOG);\n'
    assert s.count(a) == 1, f'fout pattern not found in {fn}'
    s = s.replace(a, a + '    fout << std::setprecision(17);\n')
    s = '#include <iomanip>\n' + s
    open(fn, 'w').write(s); print(f'testbench prints 17 digits: {fn}', flush=True)
    return True


def cosim_no_trace(odir):
    fn = os.path.join(odir, 'build_prj.tcl')
    s = open(fn).read()
    if '-trace_level all' in s:
        open(fn, 'w').write(s.replace('-trace_level all', '-trace_level none'))
        return True
    return False


def raise_fd_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else 1 << 20
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    print(f'RLIMIT_NOFILE {soft} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}', flush=True)


def apply_all(odir, tag, programmable=False):
    """Every patch, idempotent. Call after hm.write(), before hm._compile() / hm.build()."""
    raise_fd_limit()
    patch_avgpool(odir)
    patch_hard_tanh(odir)
    if programmable:
        unpin_multipliers(odir)
    patch_tb_precision(odir, tag)
    cosim_no_trace(odir)
