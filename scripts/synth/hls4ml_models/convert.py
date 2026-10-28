"""Baseline codes bundle -> hls4ml C++ parity and optional Vitis csynth.

convert.py TAG BUNDLE {predict,csynth} [--programmable] [--out PROJECT]
Uses validated RF1/native scheduling; no Vitis cosimulation or ASIC work.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from export import sha


def expose_weights(project, tag):
    cpp = project/'firmware'/f'{tag}.cpp'
    src = cpp.read_text()
    signature = src[src.index(f'void {tag}('):src.index(') {')]
    arrays = [(t,n,l) for t,n,l in re.findall(r'(\w+)\s+(\w+)\[(\w+)\]',signature)
              if re.fullmatch(r'[wb]\d+',n)]
    if not arrays:
        raise ValueError('BramFactor0 exposed no parameter ports')
    src = re.sub(r'[ \t]*#pragma HLS INTERFACE bram port=.*\n','',src)
    native = '    #pragma HLS DATAFLOW\n' if '    #pragma HLS DATAFLOW\n' in src else '    #pragma HLS PIPELINE\n'
    if src.count(native) != 1:
        raise ValueError('Expected native HLS top pragma')
    src = src.replace(native,''.join(f'    #pragma HLS ARRAY_RESHAPE variable={n} complete dim=0\n'
                      f'    #pragma HLS INTERFACE ap_stable port={n}\n' for _,n,_ in arrays)+native)
    begin = src.index('#ifndef __SYNTHESIS__\n    static bool loaded_weights')
    end = src.index('#endif',begin)+len('#endif\n')
    loads = re.findall(r'nnet::load_weights_from_txt<.*?;',src[begin:end])
    if len(loads) != len(arrays):
        raise ValueError('Parameter loader count mismatch')
    cpp.write_text(src[:begin]+src[end:])
    loader = '\nstatic void load_all_weights() {\n    static bool loaded = false;\n    if (loaded) return;\n'+''.join(f'    {line}\n' for line in loads)+'    loaded = true;\n}\n'
    for name,call in [(f'{tag}_test.cpp',f'{tag}(codes,'),(f'{tag}_bridge.cpp',f'{tag}(codes_ap,')]:
        path = project/name; text = path.read_text()
        pos = text.index('\n',text.rindex('#include "firmware/weights/'))+1
        if call not in text:
            raise ValueError('Generated caller pattern missing: '+name)
        path.write_text((text[:pos]+loader+text[pos:]).replace(call,'load_all_weights(); '+call))
    definitions = (project/'firmware/defines.h').read_text()+(project/'firmware/parameters.h').read_text()
    ports = []
    for typ,name,length in arrays:
        fixed = re.search(r'typedef\s+ap_(u?)fixed<\s*(\d+)\s*,\s*(-?\d+)[^>]*>\s+'+typ+r'\s*;',definitions)
        integer = re.search(r'typedef\s+ap_(u?)int<\s*(\d+)\s*>\s+'+typ+r'\s*;',definitions)
        if fixed:
            width, ibits = int(fixed[2]),int(fixed[3]); typename=f'{fixed[1]}fixed<{width},{ibits}>'
        elif integer:
            width=int(integer[2]);typename=f'{integer[1]}int<{width}>'
        else:
            raise ValueError('Unknown parameter type: '+typ)
        n = int(length) if length.isdigit() else int(re.search(rf'\b{length}\s*=\s*(\d+)',definitions)[1])
        ports.append(dict(layer=typ,name=name,n=n,width=width,type=typename,bits=n*width))
    return ports


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tag');ap.add_argument('bundle',type=Path);ap.add_argument('mode',choices=['predict','csynth'])
    ap.add_argument('--programmable',action='store_true');ap.add_argument('--out',type=Path)
    ap.add_argument('--tb-events',type=int,default=2000)
    ap.add_argument('--part',default='xcu250-figd2104-2L-e');ap.add_argument('--clock-ns',type=float,default=25.)
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',args.tag):raise ValueError('Tag must be a C identifier')
    project=(args.out or Path('prj_'+args.tag)).resolve()
    if project.exists():raise FileExistsError('Use a fresh HLS project: '+str(project))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TF_USE_LEGACY_KERAS'] = '0'  # QKeras uses standalone Keras2, including subprocesses
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):os.environ[key]='1'
    import numpy as np
    import tensorflow as tf
    from qkeras.utils import _add_supported_quantized_objects
    import hls4ml
    common=Path(__file__).resolve().parents[1]/'hls4ml_common';sys.path.insert(0,str(common))
    import hls4ml_fixes,cosim_check
    bundle=args.bundle.resolve();manifest=json.loads((bundle.parent/'export_manifest.json').read_text())
    if sha(bundle)!=manifest['bundle_sha256']:raise ValueError('Bundle hash does not match export manifest')
    model_file=bundle.parent/manifest['model_file']
    if sha(model_file)!=manifest['model_sha256']:raise ValueError('Codes model hash mismatch')
    co={};_add_supported_quantized_objects(co)
    model=tf.keras.models.load_model(model_file,custom_objects=co,compile=False)
    with np.load(bundle,allow_pickle=False) as z:x=z['codes'].astype(np.float64);y=z['keras']
    if x.shape!=(len(y),16,16,2) or y.shape!=(len(x),14) or len(x)!=manifest['validation_events']:
        raise ValueError('Unexpected full-cache shape/count')
    if not np.isfinite(y).all() or not np.isin(x,[0,1,2,3]).all():raise ValueError('Invalid vectors')
    if not 0<args.tb_events<=len(x):raise ValueError('tb-events must fit available validation data')
    for j in range(0,len(x),1000):np.testing.assert_array_equal(model(x[j:j+1000],training=False).numpy(),y[j:j+1000])
    cfg=hls4ml.utils.config_from_keras_model(model,granularity='name',backend='Vitis',default_precision='ap_fixed<32,16>')
    cfg['Model'].update(ReuseFactor=1,Strategy='Latency',BramFactor=0 if args.programmable else 10**9)
    cfg['LayerName'][model.layers[0].name]['Precision']['result']='ap_ufixed<2,2>'
    is_mlp=manifest['model']=='QMlp_Max'
    for name,lc in cfg['LayerName'].items():
        lc.update(ReuseFactor=1,Strategy='Latency',Trace=False)
        p=lc.get('Precision')
        if isinstance(p,dict) and 'accum' in p and p['accum']!='auto':p['accum']='auto'
        if 'average_pooling' in name:
            lc['Precision']=dict(accum='ap_fixed<11,7>' if is_mlp else 'ap_fixed<24,6>',
                                 result='ap_fixed<7,3>' if is_mlp else 'ap_fixed<24,6>')
        if 'conv2d' in name:lc['ParallelizationFactor']=196
    hm=hls4ml.converters.convert_from_keras_model(model,hls_config=cfg,output_dir=str(project),backend='Vitis',
         part=args.part,clock_period=args.clock_ns,io_type='io_parallel',project_name=args.tag)
    hm.write();hls4ml_fixes.apply_all(str(project),args.tag,programmable=args.programmable)
    ports=expose_weights(project,args.tag) if args.programmable else []
    (project/'ports.json').write_text(json.dumps(ports,indent=2)+'\n')
    hm._compile()  # compile() would rewrite the patched sources
    predicted=np.asarray(hm.predict(np.ascontiguousarray(x))).reshape(y.shape)
    np.testing.assert_array_equal(predicted,y)
    checks=dict(n=len(x),exact=len(x),max_err=0.,bundle_sha256=sha(bundle),
                scope='all supplied cache events compiled hls4ml C++; not RTL',model=manifest['model'])
    (project/'csim_check.json').write_text(json.dumps(checks,indent=2)+'\n')
    data=project/'tb_data';data.mkdir(exist_ok=True)
    np.savetxt(data/'tb_input_features.dat',x[:args.tb_events].reshape(args.tb_events,-1),fmt='%d')
    np.savetxt(data/'tb_output_predictions.dat',y[:args.tb_events],fmt='%.17g')
    binary=project/'tbcheck.exe'
    subprocess.run(['g++','-O1','-std=c++14','-Ifirmware/ap_types',f'-DWEIGHTS_DIR="{project}/firmware/weights"',
                    f'{args.tag}_test.cpp',f'firmware/{args.tag}.cpp','-o',str(binary)],cwd=project,check=True)
    with (project/'tbcheck.log').open('w') as log:subprocess.run([str(binary)],cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
    tb=cosim_check.check_project(str(project),y[:args.tb_events],args.tb_events,kinds=('csim',),json_out=str(project/'tbcheck.json'))
    if not tb['csim']['ok']:raise ValueError('C++ testbench differs from reference')
    binary.unlink()
    run=dict(tag=args.tag,programmable=args.programmable,model=manifest['model'],bundle_sha256=sha(bundle),
             export_provenance=manifest,part=args.part,clock_period_ns=args.clock_ns,
             tb_events=args.tb_events,compiled_events=len(x),config_payload_bits=sum(p['bits'] for p in ports),
             scripts_sha256={p.name:sha(p) for p in [Path(__file__),*common.glob('*.py')]})
    (project/'conversion.json').write_text(json.dumps(run,indent=2)+'\n')
    if args.mode=='csynth':
        report=hm.build(reset=False,csim=False,synth=True,cosim=False,validation=False,export=False,vsynth=False)
        (project/'hls_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
