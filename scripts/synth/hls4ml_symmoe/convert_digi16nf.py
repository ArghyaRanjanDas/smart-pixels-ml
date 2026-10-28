"""Portable full digi16-nf conversion, built from committed d02d428.

  convert_digi16nf.py shiqi_moe_const VAL.npz [predict|csynth] --work NEW_DIR
  convert_digi16nf.py shiqi_moe_prog  VAL.npz [predict|csynth] --work NEW_DIR

Default predict performs compiled C++ checks over every bundled event and the
standalone g++ testbench over min(2000,N). csynth additionally invokes Vitis HLS.
Fresh work/project directories are mandatory. The tag suffix selects fixed
constant coefficients or programmable coefficient ports. Precision/settings:
Vitis backend, xcu250-figd2104-2L-e,25ns,io_parallel,Latency,RF1.
Optional environment: N_TBCHECK (2000),WPORT_IF (ap_stable). PROG, if set, must
agree with the tag. This supports the four-expert full digi16-nf model only.
"""
import sys, os, json, re, subprocess, argparse, hashlib
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description='Full digi16-nf HLS conversion; all validation events are checked')
parser.add_argument('tag', help='C identifier, e.g. shiqi_moe_const or shiqi_moe_prog')
parser.add_argument('bundle', type=Path)
parser.add_argument('mode', choices=['predict','csynth'], nargs='?', default='predict')
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--common', type=Path, default=Path(__file__).resolve().parent.parent/'hls4ml_common')
args = parser.parse_args()
assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.tag), 'tag must be a C identifier'
assert args.tag.endswith(('_const','_prog')), 'tag must end _const or _prog'
tag, vnpz, mode = args.tag, args.bundle.resolve(), args.mode
common = args.common.resolve()
work = args.work.resolve()
assert not (work/'run_config.json').exists() and not (work/('prj_'+tag)).exists(), 'Use a fresh --work directory; preserve previous build evidence'
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0,HERE); sys.path.insert(0,str(common))
import hls4ml
import hls4ml_fixes, cosim_check
import symmoe_nf_hls4ml as S
S.register('Vitis')
PROG = tag.endswith('_prog')
if 'PROG' in os.environ:
    assert (os.environ['PROG']=='1') == PROG, 'PROG must agree with tag suffix'
N_TBCHECK = int(os.environ.get('N_TBCHECK','2000'))
assert N_TBCHECK > 0
WPORT_IF = os.environ.get('WPORT_IF','ap_stable')

z = np.load(vnpz, allow_pickle=False)
X = z['codes'].astype(np.float64)                 # (N,16,16,2) 2-bit codes
Y = z['y_fixed']                                  # bit-exact fixed-point reference, on the 2^-FO grid
hw = json.loads(str(z['hw_json']))
cfg, fm = hw['cfg'], {k: tuple(v) for k, v in hw['formats'].items()}
Q = {p: z['q::' + p] for p in S.PNAMES}
assert X.ndim == 4 and X.shape[1:] == (16,16,2) and len(X) > 0
assert np.isin(X,[0,1,2,3]).all() and Y.shape == (len(X),14) and np.isfinite(Y).all()
N_TBCHECK = min(N_TBCHECK,len(X))
metadata = json.loads(str(z['metadata_json'])) if 'metadata_json' in z else {'provenance_scope':'legacy bundle; checkpoint identity unavailable'}
run = dict(tag=tag,mode=mode,programmable=PROG,bundle=str(vnpz),bundle_sha256=hashlib.sha256(vnpz.read_bytes()).hexdigest(),
           events=len(X),c_testbench_events=N_TBCHECK,metadata=metadata,
           source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(HERE).glob('*') if p.suffix in ['.py','.h']})
Path('run_config.json').write_text(json.dumps(run,indent=2)+'\n')
m = S.build_keras_model(Q, fm, cfg)

hcfg = hls4ml.utils.config_from_keras_model(m, granularity='name', backend='Vitis', default_precision='ap_fixed<32,16>')
hcfg['Model']['ReuseFactor'] = 1; hcfg['Model']['Strategy'] = 'Latency'
hcfg['Model']['BramFactor'] = 0 if PROG else 1_000_000_000
hcfg['LayerName']['codes']['Precision']['result'] = 'ap_ufixed<2,2>'
for name, lc in hcfg['LayerName'].items():
    lc['ReuseFactor'] = 1; lc['Strategy'] = 'Latency'; lc['Trace'] = False
    if name == 'symmoe_nf':
        lc.pop('Precision', None)
print(json.dumps(hcfg, indent=1, default=str))
odir = os.path.abspath(f'prj_{tag}')
hm = hls4ml.converters.convert_from_keras_model(m, hls_config=hcfg, output_dir=odir, backend='Vitis',
        part='xcu250-figd2104-2L-e', clock_period=25, io_type='io_parallel', project_name=tag)
hm.write()
hls4ml_fixes.apply_all(odir, tag, programmable=PROG)

cpp = f'{odir}/firmware/{tag}.cpp'
src = open(cpp).read()
sig = src[src.index(f'void {tag}('):src.index(') {')]
wports = [(t, n, l) for t, n, l in re.findall(r'(\w+)\s+(\w+)\[(\w+)\]', sig) if re.match(r'^s\d+_\w+$', n)]
print('weight ports:', [(n, l) for _, n, l in wports])
if PROG:
    assert len(wports) == len(S.PNAMES), 'BramFactor=0 did not expose all 21 parameter arrays'
    nb = len(re.findall(r'#pragma HLS INTERFACE bram port=.*\n', src))
    src = re.sub(r'[ \t]*#pragma HLS INTERFACE bram port=.*\n', '', src); print('removed bram interface pragmas:', nb)
    native = '    #pragma HLS PIPELINE\n'
    assert src.count(native) == 1, 'expected hls4ml PIPELINE top'
    pr = ''.join(f'    #pragma HLS ARRAY_RESHAPE variable={n} complete dim=0\n    #pragma HLS INTERFACE {WPORT_IF} port={n}\n'
                 for _, n, _ in wports)
    src = src.replace(native, pr + native)
    i0 = src.index('#ifndef __SYNTHESIS__\n    static bool loaded_weights'); i1 = src.index('#endif', i0) + len('#endif\n')
    loads = re.findall(r'nnet::load_weights_from_txt<.*?;', src[i0:i1]); assert len(loads) == len(wports)
    src = src[:i0] + src[i1:]
    open(cpp, 'w').write(src)
    loader = ('\nstatic void load_all_weights() {\n    static bool loaded = false;\n    if (loaded) return;\n'
              + ''.join(f'    {l}\n' for l in loads) + '    loaded = true;\n}\n')
    for fn, call in ((f'{odir}/{tag}_test.cpp', f'{tag}(codes,'), (f'{odir}/{tag}_bridge.cpp', f'{tag}(codes_ap,')):
        t = open(fn).read()
        last_inc = t.rindex('#include "firmware/weights/'); e = t.index('\n', last_inc) + 1
        t = t[:e] + loader + t[e:]
        n = t.count(call); assert n >= 1, (fn, call)
        t = t.replace(call, 'load_all_weights(); ' + call)
        open(fn, 'w').write(t); print(f'{fn}: loader + {n} calls')
    D = open(f'{odir}/firmware/defines.h').read() + open(f'{odir}/firmware/parameters.h').read()
    ports = []
    for t, n, L in wports:
        mm = re.search(r'typedef\s+ap_(u?)fixed<\s*(\d+)\s*,\s*(-?\d+)[^>]*>\s+' + t + r'\s*;', D)
        assert mm, f'type {t} not found'
        Wd, I = int(mm.group(2)), int(mm.group(3))
        p = n.split('_', 1)[1]
        assert (Wd, I) == tuple(fm[p]), (n, Wd, I, fm[p])
        L = int(L) if L.isdigit() else int(re.search(rf'\b{L}\s*=\s*(\d+)', D).group(1))
        ports.append(dict(layer=t, name=n, reference_key='q::'+p, n=L, width=Wd, type=f'{mm.group(1)}fixed<{Wd},{I}>', bits=Wd * L))
    json.dump(ports, open(f'{tag}_ports.json', 'w'), indent=1)
    print(f'[{tag}] programmable bits: {sum(p["bits"] for p in ports)}')
else:
    assert not wports, 'fixed build must not expose parameter ports'

hm._compile()
P = np.asarray(hm.predict(np.ascontiguousarray(X.reshape(len(X), -1)))).reshape(Y.shape)
err = np.abs(P - Y)
nex = int(np.all(err == 0, 1).sum())
print(f'[{tag}] C-sim vs fixed-point reference on {len(X)} events: exact {nex}/{len(X)}, max abs err {err.max():.3g}',
      flush=True)
json.dump(dict(n=len(X), exact=nex, max_err=float(err.max()), bundle_sha256=run['bundle_sha256'],
               integer_reference_sha256=run['source_sha256']['symmoe_nf_ref.py'],
               kernel_sha256=hashlib.sha256(Path(odir,'firmware/nnet_utils/nnet_symmoe_nf.h').read_bytes()).hexdigest()),
          open(f'{tag}_csim_check.json', 'w'))
assert err.max() == 0


def write_tb(n):
    np.savetxt(f'{odir}/tb_data/tb_input_features.dat', X[:n].reshape(n, -1), fmt='%d')
    np.savetxt(f'{odir}/tb_data/tb_output_predictions.dat', Y[:n], fmt='%.17g')


os.makedirs(f'{odir}/tb_data', exist_ok=True)
write_tb(N_TBCHECK)
subprocess.run(['g++','-O1','-std=c++14','-Ifirmware/ap_types',
                '-DWEIGHTS_DIR="'+odir+'/firmware/weights"', tag+'_test.cpp','firmware/'+tag+'.cpp',
                '-o','tbcheck.exe'],cwd=odir,check=True)
with open(Path(odir)/'tbcheck.log','w') as log:
    r = subprocess.run([str(Path(odir)/'tbcheck.exe')],cwd=odir,stdout=log,stderr=subprocess.STDOUT)
assert r.returncode == 0, 'g++ testbench build/run failed'
tbres = cosim_check.check_project(odir, Y[:N_TBCHECK], N_TBCHECK, kinds=('csim',), json_out=f'{tag}_tbcheck.json')
assert tbres['csim']['ok'], 'C testbench != reference'
os.remove(f'{odir}/tbcheck.exe')
if mode == 'csynth':
    rep = hm.build(reset=False, csim=False, synth=True, cosim=False, validation=False, export=False, vsynth=False)
    json.dump(rep, open(f'{tag}_report.json', 'w'), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
