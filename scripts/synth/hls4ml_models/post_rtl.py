"""Verify baseline synthesized RTL and export a self-contained fixed/prog bundle.

post_rtl.py TAG PROJECT BUNDLE --out RTL_EXPORT [--programmable]
Compiled C++ covers all supplied cache events; RTL defaults to200events per pass.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from export import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tag');ap.add_argument('project',type=Path);ap.add_argument('bundle',type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--programmable',action='store_true')
    ap.add_argument('--rtl-events',type=int,default=200);ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',args.tag):raise ValueError('Invalid tag')
    project,bundle,out=args.project.resolve(),args.bundle.resolve(),args.out.resolve()
    proofdir=project/'post_rtl'
    if out.exists() or proofdir.exists():raise FileExistsError('Preserve existing RTL export/proof; use a fresh project/output')
    import numpy as np
    common=Path(__file__).resolve().parents[1]/'hls4ml_common';sys.path.insert(0,str(common))
    import cosim_check
    conversion=json.loads((project/'conversion.json').read_text())
    digest=sha(bundle)
    if conversion['tag']!=args.tag or conversion['programmable']!=args.programmable or conversion['bundle_sha256']!=digest:
        raise ValueError('Conversion tag/mode/bundle mismatch')
    csim=json.loads((project/'csim_check.json').read_text())
    with np.load(bundle,allow_pickle=False) as data:x,y=data['codes'],data['keras']
    if csim.get('bundle_sha256')!=digest or csim.get('n')!=len(x) or csim.get('exact')!=len(x) or csim.get('max_err')!=0:
        raise ValueError('Full-cache compiled parity missing')
    if y.shape!=(len(x),14) or not 0<args.rtl_events<=len(x):raise ValueError('Invalid output shape or RTL event count')
    tb_count=conversion['tb_events']
    tb=cosim_check.check_project(str(project),y[:tb_count],tb_count,kinds=('csim',))['csim']
    if not tb['ok'] or tb['n_exact_events']!=tb_count:raise ValueError('C++ testbench parity missing')
    rtl=project/f'{args.tag}_prj/solution1/syn/verilog'
    if not (rtl/f'{args.tag}.v').is_file():raise FileNotFoundError('Run converter csynth before post_rtl')
    proofdir.mkdir();np.save(proofdir/'x.npy',x[:args.rtl_events]);np.save(proofdir/'y.npy',y[:args.rtl_events])
    def execute(command,name,check=True):
        with (proofdir/name).open('w') as log:
            return subprocess.run([str(v) for v in command],stdout=log,stderr=subprocess.STDOUT,check=check).returncode
    extra=[];cfg=None
    if args.programmable:
        execute([sys.executable,common/'make_cfgwrap.py',project,args.tag,project/'ports.json',proofdir/'wrapper'],'wrapper.log')
        cfg=json.loads((proofdir/'wrapper/cfg_map.json').read_text())
        if cfg['n_weight_bits']!=conversion['config_payload_bits']:raise ValueError('Configuration payload mismatch')
        extra=['--extra',proofdir/'wrapper'/f'{args.tag}_cfgwrap.v','--cfg-hex',proofdir/'wrapper'/f'{args.tag}_cfg.hex']
    top=args.tag+'_cfgwrap' if args.programmable else args.tag
    command=[sys.executable,common/'verilator_check.py','--rtl',rtl,'--top',top,'--prj',project,
             '--x',proofdir/'x.npy','--y',proofdir/'y.npy','--n',args.rtl_events,'--out',proofdir/'verilator',
             '--jobs',args.jobs]+extra
    execute(command,'verilator.log')
    vc=json.loads((proofdir/'verilator/verilator_check.json').read_text())
    for mode in ('seq','stream'):
        if not vc.get('ok') or not vc[mode]['ok'] or vc[mode]['n_exact_events']!=args.rtl_events:
            raise ValueError('RTL parity failed: '+mode)
    negative=None
    if args.programmable:
        zero=proofdir/'zero_cfg.hex';zero.write_text('00\n'*cfg['n_bytes'])
        cmd=command.copy();cmd[cmd.index('--cfg-hex')+1]=zero;cmd[cmd.index('--out')+1]=proofdir/'negative'
        rc=execute(cmd,'negative.log',False)
        negative=json.loads((proofdir/'negative/verilator_check.json').read_text());negative['returncode']=rc
        for mode in ('seq','stream'):
            if negative[mode]['n_results']!=args.rtl_events or negative[mode]['n_exact_events']==args.rtl_events:
                raise ValueError('Zero configuration did not demonstrate parameter dependence')
    out.mkdir(parents=True)
    for path in rtl.iterdir():
        if path.suffix in ('.v','.dat','.mem'):shutil.copy2(path,out/path.name)
    if args.programmable:
        for name in (f'{args.tag}_cfgwrap.v',f'{args.tag}_cfg.hex','cfg_map.json'):
            shutil.copy2(proofdir/'wrapper'/name,out/name)
    for path in [project/'conversion.json',project/'csim_check.json',bundle.parent/'export_manifest.json',
                 proofdir/'verilator/verilator_check.json']:
        shutil.copy2(path,out/path.name)
    for path in (project/f'{args.tag}_prj/solution1/syn/report').glob(f'{args.tag}_csynth.*'):
        shutil.copy2(path,out/path.name)
    timing=vc['timing']
    report_path=project/f'{args.tag}_prj/solution1/syn/report/{args.tag}_csynth.rpt'
    totals=re.search(r'\|Total\s*\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|',report_path.read_text())
    if not totals:raise ValueError('Missing HLS resource totals')
    resources=dict(zip(['BRAM_18K','DSP','FF','LUT','URAM'],totals.groups()))
    manifest=dict(design=args.tag,top=top,core=args.tag,model=conversion['model'],programmable=args.programmable,
                  clock_port='ap_clk',clock_period_ns=conversion['clock_period_ns'],reset_port='ap_rst',
                  reset_polarity='active-high synchronous',latency_clk=[timing['latency_min'],timing['latency_max']],
                  ii=timing['ii_ready_max'],result_type=vc['result_t'],outputs=14,
                  csynth=dict(total=resources,scope='FPGA HLS core estimates, excluding wrapper FF; not routed ASIC'),
                  export_provenance=conversion['export_provenance'],
                  verification=dict(compiled_c=csim,c_testbench=tb,rtl=vc,negative_control=negative),
                  validation_scope=dict(compiled_events=len(x),gxx_events=tb_count,rtl_events_per_pass=args.rtl_events,
                                        vitis_cosim='not run; Verilator sequential and streaming instead'),
                  files=sorted(p.name for p in out.glob('*.v')),source_sha256={str(p):sha(p) for p in [Path(__file__),*common.glob('*.py')]})
    if cfg:manifest['cfg']=dict(n_cfg_ffs=cfg['n_cfg_bits'],n_weight_bits=cfg['n_weight_bits'],n_bytes=cfg['n_bytes'],
                               image=f'{args.tag}_cfg.hex',map='cfg_map.json')
    manifest['file_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(ok=True,export=str(out),rtl_events_per_pass=args.rtl_events)))


if __name__=='__main__':main()
