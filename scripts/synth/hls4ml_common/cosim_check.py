#!/usr/bin/env python
"""THE value checker for hls4ml C-sim / RTL co-simulation outputs (shared by every SmartPixels HLS design).

Why it exists: the hls4ml testbench never compares values, so Vitis prints "Pass" for any RTL that merely
runs (the avg-pool unsigned-divide bug passed that way, see patch_avgpool.py). Every result must be compared
with Keras here, on the output LSB grid, counting exactly-equal events.

Inputs it understands
  * tb_data/{csim,rtl_cosim}_results.log  -- decimal text written by nnet::print_result (one event per line).
    The stock testbench prints with 6 significant digits, so a raw float comparison flags EVERY event
    (that is what made B_lut_prog look like "csim_vs_keras 200/200 bad"). Values are snapped to the
    result_t grid (2^-F); the snapping residual is reported and must stay < 0.25 LSB, otherwise the
    print precision is too low to decide and the check fails (use hls4ml_fixes.patch_tb_precision).
  * raw-integer files (xsim wrapper testbench tb_out.txt): already in LSB units (raw=True).

Verdict `ok` = every expected event present, every value equal to Keras on the grid, unambiguous print.

  python cosim_check.py <prj_dir> <keras_out.npy|.npz[:key]> [--n N] [--kinds csim,rtl_cosim] [--json out.json]
  python cosim_check.py --raw <tb_out.txt> <keras_out.npy> --frac-bits 14 [--n N] [--json out.json]
"""
import os, re, sys, json, argparse
import numpy as np


def result_frac_bits(prj, typename='result_t'):
    """Fractional bits of result_t from <prj>/firmware/defines.h (ap_fixed<W,I> -> W-I)."""
    d = open(os.path.join(prj, 'firmware', 'defines.h')).read()
    m = re.search(r'typedef\s+ap_(u?)fixed<\s*(\d+)\s*,\s*(-?\d+)[^>]*>\s+' + typename + r'\s*;', d)
    assert m, f'{typename} (ap_fixed) not found in {prj}/firmware/defines.h'
    return int(m.group(2)) - int(m.group(3))


def _load_rows(path, ncol):
    rows = []
    for ln in open(path):
        t = ln.split()
        if t:
            rows.append([float(v) for v in t])
    if not rows:
        return np.zeros((0, ncol))
    bad = [i for i, r in enumerate(rows) if len(r) != ncol]
    assert not bad, f'{path}: line(s) {bad[:5]} do not have {ncol} values'
    return np.asarray(rows, dtype=np.float64)


def check_values(R, Y, frac_bits, n_expected=None, raw=False, label=''):
    """R: results (decimal, or raw LSB integers if raw), Y: Keras outputs (float, on the 2^-frac_bits grid)."""
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    n_expected = len(Y) if n_expected is None else int(n_expected)
    R = np.asarray(R, dtype=np.float64).reshape(len(R), -1) if len(R) else np.zeros((0, Y.shape[1]))
    s = 2.0 ** frac_bits
    Yi = Y[:n_expected] * s
    assert np.array_equal(Yi, np.rint(Yi)), f'Keras outputs are not on the 2^-{frac_bits} grid'
    Ri_f = R if raw else R * s
    Ri = np.rint(Ri_f)
    n = min(len(R), n_expected)
    resid = float(np.abs(Ri_f - Ri).max()) if len(R) else 0.0
    diff = np.abs(Ri[:n] - Yi[:n])
    exact = (diff.max(1) == 0) if n else np.zeros(0, bool)
    out = dict(label=label, n_expected=n_expected, n_results=int(len(R)), n_compared=int(n),
               n_exact_events=int(exact.sum()), n_bad_events=int(n - exact.sum()),
               n_bad_values=int((diff > 0).sum()), max_abs_err_lsb=float(diff.max()) if n else None,
               frac_bits=int(frac_bits), print_resid_lsb=resid,
               first_bad_events=[int(i) for i in np.where(~exact)[0][:10]])
    out['ok'] = bool(len(R) == n_expected and n == n_expected and exact.all() and resid < 0.25)
    if not out['ok']:
        why = []
        if len(R) != n_expected: why.append(f'{len(R)} result rows, expected {n_expected}')
        if n and not exact.all(): why.append(f'{n - exact.sum()} events differ from Keras')
        if resid >= 0.25: why.append(f'print precision too low (resid {resid:.3f} LSB)')
        out['why'] = '; '.join(why)
    return out


def check_project(prj, Y, n_expected=None, kinds=('csim', 'rtl_cosim'), json_out=None, verbose=True):
    """Check <prj>/tb_data/<kind>_results.log against Keras Y for every kind. Missing file -> ok=False."""
    F = result_frac_bits(prj)
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    res = {}
    for k in kinds:
        f = os.path.join(prj, 'tb_data', f'{k}_results.log')
        if not os.path.exists(f):
            res[k] = dict(label=k, ok=False, why=f'{f} missing', n_results=0,
                          n_expected=len(Y) if n_expected is None else int(n_expected))
        else:
            res[k] = check_values(_load_rows(f, Y.shape[1]), Y, F, n_expected, label=k)
        if verbose:
            r = res[k]
            print(f'[cosim_check] {prj} {k}: ok={r["ok"]} exact {r.get("n_exact_events", 0)}/{r["n_expected"]} '
                  f'(rows {r["n_results"]}, bad values {r.get("n_bad_values")}, max err {r.get("max_abs_err_lsb")} LSB, '
                  f'print resid {r.get("print_resid_lsb")} LSB){" -- " + r["why"] if "why" in r else ""}', flush=True)
    if json_out:
        json.dump(res, open(json_out, 'w'), indent=1)
    return res


def _load_y(spec):
    path, _, key = spec.partition(':')
    z = np.load(path)
    if path.endswith('.npz'):
        return z[key or 'keras']
    return z


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('target'); ap.add_argument('keras')
    ap.add_argument('--n', type=int); ap.add_argument('--kinds', default='csim,rtl_cosim')
    ap.add_argument('--raw', action='store_true'); ap.add_argument('--frac-bits', type=int)
    ap.add_argument('--json')
    a = ap.parse_args()
    Y = _load_y(a.keras)
    if a.raw:
        assert a.frac_bits is not None, '--raw needs --frac-bits'
        Y2 = np.asarray(Y, np.float64).reshape(len(Y), -1)
        r = check_values(_load_rows(a.target, Y2.shape[1]), Y2, a.frac_bits, a.n, raw=True, label=a.target)
        print(json.dumps(r))
        if a.json: json.dump(r, open(a.json, 'w'), indent=1)
        sys.exit(0 if r['ok'] else 1)
    res = check_project(a.target, Y, a.n, a.kinds.split(','), a.json)
    sys.exit(0 if all(r['ok'] for r in res.values()) else 1)
