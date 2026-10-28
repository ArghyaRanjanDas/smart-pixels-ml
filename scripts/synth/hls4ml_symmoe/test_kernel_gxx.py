"""Standalone g++ check of nnet_symmoe_nf.h against symmoe_nf_ref.fixed_ref (no hls4ml): every val event.

  python test_kernel_gxx.py <digi16nf_export.npz OR prepared digi16nf_val.npz> <ap_types dir> <work dir>
"""
import os, sys, subprocess, json, hashlib
from pathlib import Path
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import symmoe_nf_ref as R
import symmoe_nf_hw as H

npz, apt, wd = sys.argv[1:4]
os.makedirs(wd, exist_ok=True)
z = np.load(npz)
codes = z['codes']
assert codes.ndim == 4 and codes.shape[1:] == (16,16,2) and len(codes) > 0
assert np.isin(codes,[0,1,2,3]).all()
codes = codes.astype(np.uint8)
if 'hw_json' in z:
    hw=json.loads(str(z['hw_json']));cfg=hw['cfg'];fm={k:tuple(v) for k,v in hw['formats'].items()}
    Q={p:z['q::'+p] for p,_,_ in R.PORTS}
    np.testing.assert_array_equal(H.fixed_outputs(codes,Q,fm,cfg),z['y_fixed'])
else:
    W = {k[3:]: z[k] for k in z.files if k.startswith('w::')}
    cfg, fm, Q, err = H.hw_build(W)
if os.environ.get('STRESS'):        # random full-range port contents + adversarial code patterns
    rng = np.random.default_rng(int(os.environ['STRESS']))
    for p, shape, _ in R.PORTS:
        Wd = fm[p][0]
        Q[p] = rng.integers(-(1 << (Wd - 1)), 1 << (Wd - 1), size=shape, endpoint=False)
        if rng.random() < 0.3:       # extremes
            Q[p] = np.where(rng.random(shape) < 0.5, -(1 << (Wd - 1)), (1 << (Wd - 1)) - 1)
    n = 6000
    dens = rng.random((n, 1, 1, 1))
    cs = (rng.random((n, 16, 16, 2)) < dens) * rng.integers(0, 4, (n, 16, 16, 2))
    cs[0] = 0; cs[1] = 3; cs[2] = 0; cs[2, 0, 0, 0] = 1; cs[3] = 0; cs[3, 15, 15, 1] = 3; cs[4] = 0; cs[4, :, 0, :] = 3
    cs[5] = 0; cs[5, 0, :, 1] = 3; cs[6] = 0; cs[6, 7, 7] = [1, 1]
    codes = np.concatenate([codes[:2000], cs.astype(np.uint8)], 0)
ref = R.fixed_ref(codes, Q, fm, cfg)
tn = {p: f'ap_fixed<{fm[p][0]},{fm[p][1]}>' for p, _, _ in R.PORTS}
s = '#include "ap_fixed.h"\n#include "nnet_symmoe_nf.h"\n' + H.cpp_tables(cfg, 'tab') + H.cpp_config(cfg, tn, 'cfg_t')
for p, shape, _ in R.PORTS:
    F = fm[p][0] - fm[p][1]
    s += f'static ap_fixed<{fm[p][0]},{fm[p][1]}> P_{p}[{int(np.prod(shape))}] = {{' + \
        ', '.join(repr(float(v) * 2.0 ** -F) for v in Q[p].ravel()) + '};\n'
WO = cfg['IO'] + cfg['FO']
s += f'''
#include <cstdio>
#include <vector>
int main(int argc, char** argv) {{
    FILE* f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 512; fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(n * 512); fread(buf.data(), 1, n * 512, f); fclose(f);
    FILE* fo = fopen(argv[2], "w");
    for (long e = 0; e < n; e++) {{
        ap_ufixed<2,2> x[512]; ap_fixed<{WO},{cfg["IO"]}> y[14];
        for (int i = 0; i < 512; i++) x[i] = buf[e * 512 + i];
        nnet::symmoe_nf<ap_ufixed<2,2>, ap_fixed<{WO},{cfg["IO"]}>, cfg_t>(x, y, {", ".join("P_" + p for p, _, _ in R.PORTS)},
            tab_ln, tab_exp2, tab_tanh);
        for (int k = 0; k < 14; k++) {{ ap_int<{WO}> r; r.range({WO - 1}, 0) = y[k].range({WO - 1}, 0); fprintf(fo, k ? " %lld" : "%lld", (long long)r.to_int64()); }}
        fprintf(fo, "\\n");
    }}
    fclose(fo); return 0;
}}
'''
open(f'{wd}/tk.cpp', 'w').write(s)
codes.reshape(len(codes), -1).tofile(f'{wd}/codes.bin')
subprocess.run(['g++','-O2','-std=c++14','-I'+str(Path(apt).resolve()),'-I'+HERE,'tk.cpp','-o','tk'],cwd=wd,check=True)
subprocess.run([str(Path(wd).resolve()/'tk'),'codes.bin','out.txt'],cwd=wd,check=True)
out = np.loadtxt(f'{wd}/out.txt', dtype=np.int64).reshape(ref.shape)
d = out != ref
proof=dict(events=len(ref),exact=int((~d.any(1)).sum()),mismatching_values=int(d.sum()),
           input_sha256=hashlib.sha256(Path(npz).read_bytes()).hexdigest(),
           kernel_sha256=hashlib.sha256(Path(HERE,'nnet_symmoe_nf.h').read_bytes()).hexdigest(),
           reference_sha256=hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest())
Path(wd,'kernel_check.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
if d.any():
    e = np.where(d.any(1))[0][:5]
    for i in e:
        print(i, 'cols', np.where(d[i])[0], out[i][d[i]], ref[i][d[i]])
sys.exit(0 if not d.any() else 1)
