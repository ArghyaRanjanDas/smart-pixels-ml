#!/usr/bin/env python
"""Verify and export a full digi16-nf HLS build; no ASIC routing is performed.

--work is the converter work directory containing run_config.json and prj_TAG.
--out is a NEW/empty RTL artifact directory. Defaults use sibling common tools
and this Python interpreter; pass --python for an environment with NumPy/Verilator.
The model is the supported four-expert signed barycenter digi16-nf formula,
not a generic PySR expression compiler. No SLIM support.
"""
import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--common',type=Path,default=Path(__file__).resolve().parent.parent/'hls4ml_common')
    ap.add_argument('--python',type=Path,default=Path(sys.executable))
    ap.add_argument('--jobs',type=int,default=8)
    ap.add_argument('--events',type=int,default=200)
    a=ap.parse_args()
    work,out,common=a.work.resolve(),a.out.resolve(),a.common.resolve()
    assert a.jobs>0 and a.events>0
    assert not out.exists() or not any(out.iterdir()),'--out must be new/empty; preserve prior evidence'
    sys.path.insert(0,str(common));import cosim_check
    run=json.loads((work/'run_config.json').read_text());tag=run['tag'];prog=run['programmable']
    assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',tag) and prog==tag.endswith('_prog')
    prj=work/('prj_'+tag);vdir=prj/(tag+'_prj/solution1/syn/verilog')
    assert (vdir/(tag+'.v')).is_file(), 'No synthesized RTL; run converter csynth first'
    bundle=Path(run['bundle']);assert sha(bundle)==run['bundle_sha256'],'bundle changed since conversion'
    z=np.load(bundle,allow_pickle=False);X=z['codes'];Y=z['y_fixed'];n=len(X)
    assert n==run['events'] and Y.shape==(n,14)
    k=min(a.events,n);ntb=run['c_testbench_events']
    csim=json.loads((work/(tag+'_csim_check.json')).read_text())
    assert csim['n']==csim['exact']==n and csim['max_err']==0
    assert csim['bundle_sha256']==run['bundle_sha256'], 'C++ proof belongs to a different bundle'
    assert csim['integer_reference_sha256']==run['source_sha256']['symmoe_nf_ref.py']
    assert csim['kernel_sha256']==run['source_sha256']['nnet_symmoe_nf.h']
    assert sha(prj/'firmware/nnet_utils/nnet_symmoe_nf.h')==csim['kernel_sha256']
    ctb=cosim_check.check_project(str(prj),Y,ntb,kinds=('csim',))['csim']
    assert ctb['ok'] and ctb['n_exact_events']==ntb
    tb=work/'tb_in';tb.mkdir(exist_ok=True);np.save(tb/'x.npy',X[:k]);np.save(tb/'y.npy',Y[:k])
    rpt=prj/(tag+'_prj/solution1/syn/report')/(tag+'_csynth.rpt');r=rpt.read_text()
    latency=re.search(r'Latency \(cycles\).*?\n.*?\n.*?\n\s*\|\s*(\d+)\|\s*(\d+)\|[^|]*\|[^|]*\|\s*(\d+)\|\s*(\d+)\|',r)
    total=re.search(r'\|Total\s*\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|',r)
    assert latency and total,'Missing HLS latency/resources in report'
    res=dict(latency=[int(latency[1]),int(latency[2])],ii=[int(latency[3]),int(latency[4])],
             total=dict(zip(['BRAM_18K','DSP','FF','LUT','URAM'],total.groups())),scope='FPGA HLS estimates; not routed FPGA/ASIC')
    def vl(name,top,extra=(),image=None):
        dest=work/name
        cmd=[str(a.python),str(common/'verilator_check.py'),'--rtl',str(vdir),'--top',top,'--prj',str(prj),
             '--x',str(tb/'x.npy'),'--y',str(tb/'y.npy'),'--n',str(k),'--out',str(dest),'--jobs',str(a.jobs)]
        for p in extra:cmd+=['--extra',str(p)]
        if image:cmd+=['--cfg-hex',str(image)]
        p=subprocess.run(cmd,capture_output=True,text=True)
        (work/(name+'.log')).write_text(p.stdout+p.stderr)
        report=dest/'verilator_check.json'
        assert report.is_file(),f'No Verilator report; see {name}.log'
        return p.returncode,json.loads(report.read_text()),report
    cmap=None;neg=None;wrap=work/'wrap';top=tag
    if prog:
        wrap.mkdir(exist_ok=True)
        subprocess.run([str(a.python),str(common/'make_cfgwrap.py'),str(prj),tag,str(work/(tag+'_ports.json')),str(wrap)],check=True)
        cmap=json.loads((wrap/'cfg_map.json').read_text())
        bits=''.join(f'{int(h,16):08b}'[::-1] for h in (wrap/(tag+'_cfg.hex')).read_text().split())
        ports=json.loads((work/(tag+'_ports.json')).read_text());assert len(ports)==21
        for p in ports:
            key=p.get('reference_key','q::'+p['name'].split('_',1)[1]);q=z[key].ravel()
            lo=cmap['ports'][p['name']]['lo'];w=p['width']
            raw=[int(bits[lo+w*i:lo+w*(i+1)][::-1],2) for i in range(p['n'])]
            signed=[v-(1<<w) if v>=1<<(w-1) else v for v in raw]
            np.testing.assert_array_equal(signed,q)
        top=tag+'_cfgwrap';extra=[wrap/(top+'.v')];image=wrap/(tag+'_cfg.hex')
        rc,rtl,rtl_report=vl('vl_wrap',top,extra,image)
        zero=work/'zero_cfg.hex';zero.write_text('00\n'*cmap['n_bytes'])
        nr,neg,_=vl('vl_negctrl_zero',top,extra,zero)
        assert nr!=0 and neg['seq']['n_results']==k and neg['stream']['n_results']==k
        assert neg['seq']['n_exact_events']<k and neg['stream']['n_exact_events']<k,'Zero config did not change outputs'
    else:
        rc,rtl,rtl_report=vl('vl_core',top)
    assert rc==0 and rtl['ok'] and rtl['seq']['n_exact_events']==rtl['stream']['n_exact_events']==k
    assert not out.exists() or not any(out.iterdir()),'--out must be new/empty; previous evidence is preserved'
    out.mkdir(parents=True,exist_ok=True)
    copied=[]
    for p in sorted(vdir.glob('*')):
        if p.suffix in ['.v','.dat']:shutil.copy2(p,out/p.name);copied.append(p.name)
    shutil.copy2(rpt,out/rpt.name);shutil.copy2(rtl_report,out/'verilator_check.json')
    if prog:
        for name in [top+'.v',tag+'_cfg.hex','cfg_map.json']:shutil.copy2(wrap/name,out/name);copied.append(name)
    (out/'hw_config.json').write_text(str(z['hw_json'])+'\n')
    man=dict(design=tag,top=top,core=tag,programmable=prog,model='SymMoE_digi16nf',output_count=14,
             result_type='ap_fixed<22,8>',clock_period_ns=25,clock_port='ap_clk',reset_port='ap_rst',
             source_base_revision='d02d428 full digi16-nf; portable interface changes only',
             checkpoint=run['metadata'],bundle_sha256=run['bundle_sha256'],csynth=res,
             measured_rtl_timing=rtl['timing'],latency_clk=(rtl['timing']['latency_min'] if rtl['timing']['latency_min']==rtl['timing']['latency_max']
                          else [rtl['timing']['latency_min'],rtl['timing']['latency_max']]),
             ii=rtl['timing']['ii_ready_max'],verification=dict(compiled_cpp_all_events=csim,c_testbench=ctb,
             rtl=rtl,negative_control=neg,rtl_requested_events=a.events,rtl_actual_events=k,
             scope='Compiled C++ equals integer reference for all bundled events; RTL sequential/stream checks are separate limited scope; not full-validation RTL or formal proof'),
             reference=dict(kind='symmoe_nf_ref.fixed_ref',n_events=n,nll_float=float(z['nll_keras']),
             nll_fixed=float(z['nll_fixed']),max_abs_fixed_vs_keras=float(np.abs(Y-z['keras']).max())),
             source_sha256=run['source_sha256'],post_rtl_script_sha256=sha(__file__),
             common_sha256={p.name:sha(p) for p in [common/'cosim_check.py',common/'make_cfgwrap.py',common/'verilator_check.py']},
             files=copied,file_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
             created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if prog:man['cfg']=dict(n_cfg_ffs=cmap['n_cfg_bits'],n_weight_bits=cmap['n_weight_bits'],n_bytes=cmap['n_bytes'],image=tag+'_cfg.hex',map='cfg_map.json')
    (out/'MANIFEST.json').write_text(json.dumps(man,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(out),events=n,rtl_events=k,top=top,ok=True)))


if __name__=='__main__':main()
