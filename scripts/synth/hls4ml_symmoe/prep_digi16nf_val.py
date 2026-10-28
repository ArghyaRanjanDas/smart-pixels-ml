#!/usr/bin/env python
"""Full floating export -> bit-exact hardware bundle; use all cached events.

Usage: prep_digi16nf_val.py --export digi16nf_export.npz --out digi16nf_val.npz
Frozen port widths reject out-of-range coefficients; no silent rescaling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import symmoe_nf_ref as R
import symmoe_nf_hw as H


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--export',dest='src',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    z=np.load(a.src,allow_pickle=False)
    W={k[3:]:z[k] for k in z.files if k.startswith('w::')}
    codes=z['codes'];K=z['keras'].astype(np.float64);y=z['labels'];bsz=z['batch_sizes']
    assert codes.ndim==4 and codes.shape[1:]==(16,16,2) and len(codes)>0
    assert np.isin(codes,[0,1,2,3]).all() and K.shape==(len(codes),14) and y.shape==(len(codes),4)
    assert bsz.ndim==1 and (bsz>0).all() and sum(bsz)==len(codes)
    cfg,fm,Q,err=H.hw_build(W)
    Yf=H.fixed_outputs(codes,Q,fm,cfg);F=R.float_ref(codes,W)
    assert np.isfinite(K).all() and np.isfinite(F).all() and np.isfinite(Yf).all()
    np.testing.assert_allclose(F,K,rtol=0,atol=2e-5)
    nk=R.nll_v2(y,K,bsz)[0];nfx=R.nll_v2(y,Yf,bsz)[0]
    metadata=json.loads(str(z['metadata_json'])) if 'metadata_json' in z else {'provenance_scope':'legacy export; original checkpoint identity unavailable'}
    metadata.update(export_npz=str(a.src.resolve()),export_npz_sha256=hashlib.sha256(a.src.read_bytes()).hexdigest(),
                    events=len(codes),payload_bits=sum(fm[p][0]*int(np.prod(s)) for p,s,_ in R.PORTS),
                    integer_reference_sha256=hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(),
                    kernel_sha256=hashlib.sha256(Path(H.__file__).with_name('nnet_symmoe_nf.h').read_bytes()).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,codes=codes.astype(np.uint8),y_fixed=Yf,keras=K,labels=y,batch_sizes=bsz,
                        hw_json=H.dump_json(cfg,fm),metadata_json=json.dumps(metadata),nll_keras=nk,nll_fixed=nfx,
                        **{f'q::{p}':Q[p] for p,_,_ in R.PORTS})
    report=dict(metadata=metadata,nll_keras=nk,nll_fixed=nfx,delta=nfx-nk,
                float_reference_vs_keras_max=float(np.abs(F-K).max()),fixed_vs_keras_max=float(np.abs(Yf-K).max()),
                formats=fm,cfg=cfg,port_rounding_error=err)
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
