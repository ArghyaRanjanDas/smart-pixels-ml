"""digi16-nf hardware configuration: the fixed-point widths actually built, the frozen port formats, and the C++
config struct / ROM tables that nnet_symmoe_nf.h consumes.  Pure NumPy; no TensorFlow or hls4ml dependency.

HW_CFG was chosen with a one-knob-at-a-time sweep of symmoe_nf_ref.fixed_ref on ALL 37,919 val events
(seed 50842 best, epoch 9772), keeping each knob where its val_plain_nll change stayed within the
+-1 run-to-run noise of the sweep, then checked jointly: fixed-point val_plain_nll -16684.787 vs float Keras
-16685.469 (+0.68, 0.004 %), max |out - Keras| 0.012 (normalised label units).

Port formats (W, I) are FROZEN in PORT_FORMATS: integer bits = what the trained snapshot needs x2 headroom
(symmoe_nf_ref.port_formats), widths from the sweep.  For the programmable design they are part of the
hardware; for the fixed design the same formats are used so both builds compute the same function.
"""
import json
import math
import numpy as np

import symmoe_nf_ref as R

HW_CFG = dict(R.DEFAULT_CFG)
HW_CFG.update(FR=12, FM=12, FZ=9, FU=8, TE=4, FE=13, FW=14, FT=9, FA=7, TR=6, FS=14, FMU=13, FEH=9, LT=4, FL=14)
# (width, integer bits incl. sign) of every parameter port
PORT_FORMATS = {
    'w_r1': (15, 4), 'b_r1': (11, 2), 'w_r2': (14, 3), 'b_r2': (10, 1),
    'a_sx': (10, 2), 'a_bx': (12, 4), 'a_sy': (11, 3), 'a_by': (15, 8),
    'a_ka': (10, 5), 'a_ca': (10, 4), 'a_kb': (10, 5), 'a_cb': (10, 3),
    'a_sa': (13, 1), 'a_ba': (11, 1), 'a_sb': (11, 1), 'a_lb': (11, 3), 'a_bb': (11, 1),
    'w_e1': (14, 6), 'b_e1': (13, 4), 'w_e2': (15, 3), 'b_e2': (13, 3),
}
L2E = round(math.log2(math.e) * 2 ** 16)


def check_cfg(cfg):
    assert cfg['FA'] > cfg['TT'] and cfg['FEH'] > cfg['TT'], 'tanh: input frac must exceed the table step bits'
    assert cfg['FU'] > cfg['TE'] and 10 > cfg['LT'] and cfg['FZ'] + 16 - cfg['FU'] >= 0
    assert cfg['FM'] >= 3 and cfg['FR'] >= 1


def hw_build(W, cfg=None, fm=None):
    """trainable weights (export npz dict) -> (cfg, fm, Q ints)."""
    cfg = dict(HW_CFG if cfg is None else cfg)
    fm = dict(PORT_FORMATS if fm is None else fm)
    check_cfg(cfg)
    Pf = R.fold_params(W)
    Q, err = R.quantise_ports(Pf, fm)        # asserts every value fits its port
    return cfg, fm, Q, err


def tables(cfg):
    T = R.tables(cfg)
    out = {}
    for k in ('ln', 'exp2', 'tanh'):
        v = T[k]
        assert v.min() >= 0 and np.all(np.diff(v) >= 0), k        # increasing, unsigned (interp relies on it)
        out[k] = (v, int(v.max()).bit_length())
    return out


def cpp_tables(cfg, prefix):
    s = ''
    for k, (v, w) in tables(cfg).items():
        s += f'static const ap_uint<{w}> {prefix}_{k}[{len(v)}] = {{' + ', '.join(str(int(x)) for x in v) + '};\n'
    return s


def cpp_config(cfg, typenames, name, base='nnet::symmoe_nf_config'):
    """struct <name> with the constants and typedefs nnet::symmoe_nf<> reads. typenames: port -> C++ type name."""
    T = tables(cfg)
    keys = ['FR', 'FM', 'FH', 'IH', 'FZ', 'IZ', 'FU', 'TE', 'FE', 'FW', 'FT', 'FA', 'IA', 'TT', 'TR', 'FS', 'FMU', 'FO',
            'IO', 'FEH', 'IEH', 'LT', 'FL']
    s = f'struct {name} : {base} {{\n'
    for k in keys:
        s += f'    static const int {k} = {int(cfg[k])};\n'
    s += f'    static const int L2E = {L2E};\n'
    s += f'    static const int LN2_FL = {round(R.LN2 * 2 ** cfg["FL"])};\n'
    for p, _, _ in R.PORTS:
        s += f'    typedef {typenames[p]} {p}_t;\n'
    for k, (v, w) in T.items():
        s += f'    typedef ap_uint<{w}> {k}_t;\n'
    s += '};\n'
    return s


def fixed_outputs(codes, Q, fm, cfg):
    """bit-exact reference outputs as float on the 2^-FO grid."""
    return R.fixed_ref(codes, Q, fm, cfg).astype(np.float64) * 2.0 ** -cfg['FO']


def ports_float(Q, fm):
    """port values as floats on their grids (what hls4ml writes into firmware/weights/*.txt)."""
    return {p: Q[p].astype(np.float64) * 2.0 ** -(fm[p][0] - fm[p][1]) for p, _, _ in R.PORTS}


def dump_json(cfg, fm):
    return json.dumps(dict(cfg=cfg, formats={k: list(v) for k, v in fm.items()}))
