"""Work around a Vitis HLS 2024.1 mis-synthesis in hls4ml's average pooling (found 2026-10-19, design A_const).

hls4ml nnet_pooling.h avg():   accum_t y = ...; y /= length;   (accum_t signed ap_fixed, length `unsigned`)
C-sim / hls_model.predict divide signed (bit-exact vs Keras), but the RTL Vitis generates performs an UNSIGNED
divide of the wrapped accumulator: negative pool sums come out as (2^W + sum)/N (e.g. -2.25/9 -> +28.19).
Every stage up to the pooling was bit-exact in RTL (xsim probe of the dataflow FIFOs); every negative average was wrong.

Fix: divide the magnitude, restore the sign. ap_fixed division truncates toward zero, so this is bit-identical to the
original C (checked exhaustively for 9 x ap_fixed<4,1,RND_CONV,SAT_SYM> into ap_fixed<24,6>, all lengths 1..9).
Call patch_avgpool(output_dir) AFTER hls_model.write() and BEFORE hls_model._compile()/build().
"""
import os

OLD = """    y /= length;
    return y;
}"""
NEW = """    // patched (smartpix): Vitis HLS 2024.1 synthesizes signed-ap_fixed / unsigned as an unsigned divide.
    ap_fixed<accum_t::width + 1, accum_t::iwidth + 1> s = y;
    bool neg = s < 0;
    ap_ufixed<accum_t::width, accum_t::iwidth> a = neg ? decltype(s)(-s) : s;
    a /= length;
    y = neg ? accum_t(-a) : accum_t(a);
    return y;
}"""


def patch_avgpool(odir):
    fn = os.path.join(odir, 'firmware', 'nnet_utils', 'nnet_pooling.h')
    src = open(fn).read()
    if 'patched (smartpix)' in src:
        return False
    assert src.count(OLD) == 1, f'avg() pattern not found exactly once in {fn}'
    open(fn, 'w').write(src.replace(OLD, NEW))
    print(f'patched avg-pool signed division in {fn}', flush=True)
    return True
