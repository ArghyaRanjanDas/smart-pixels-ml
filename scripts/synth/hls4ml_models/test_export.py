"""Synthetic-only smoke check: both baseline factories, saved weights and full-cache export.

No real dataset is subsetted, no training, HLS or synthesis is run.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TF_USE_LEGACY_KERAS'] = '0'  # QKeras uses standalone Keras2, including subprocesses
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):os.environ[key]='1'
    import numpy as np
    import tensorflow as tf
    repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo/'two_bit_optimization_helpers'))
    from models.mlp_encoder_model_quantized import QMlp_Max_SoftQuantizer
    from models.conv2d_model_quantized import QConv2D_Max_SoftQuantizer
    with tempfile.TemporaryDirectory(prefix='baseline_export_smoke_',dir=repo) as temporary:
        temp=Path(temporary);rng=np.random.default_rng(42)
        cache=temp/'synthetic.npz'
        np.savez(cache,n=2,X0=rng.uniform(0,70,(3,16,16,2)).astype('float32'),y0=np.zeros((3,4),'float32'),
                 X1=rng.uniform(0,70,(2,16,16,2)).astype('float32'),y1=np.zeros((2,4),'float32'))
        for name,factory in [('QMlp_Max',lambda:QMlp_Max_SoftQuantizer((16,16,2),[8.,20.,47.],0.,np.arange(4,dtype='float32'))),
                             ('QConv2D_Max',lambda:QConv2D_Max_SoftQuantizer((16,16,2),5,3,[8.,20.,47.],0.,np.arange(4,dtype='float32')))]:
            tf.keras.backend.clear_session();model=factory();weights=temp/(name+'.weights.hdf5')
            model.get_layer('soft_quantizer_output').threshold_deltas_raw.assign(np.log1p([7.,12.,27.]))
            expected_thresholds=model.get_layer('soft_quantizer_output').thresholds.numpy()
            model.save_weights(str(weights));out=temp/name
            subprocess.run([sys.executable,str(Path(__file__).with_name('export.py')),'--model',name,
                            '--weights',str(weights),'--validation-cache',str(cache),'--out',str(out)],check=True)
            report=json.loads((out/'export_manifest.json').read_text())
            assert report['validation_events']==5 and report['original_batch_sizes']==[3,2]
            np.testing.assert_array_equal(report['thresholds'],expected_thresholds)
            assert report['full_vs_codes_only_exact_events']==5
            with np.load(out/'bundle.npz',allow_pickle=False) as z:
                assert z['keras'].shape==(5,14) and z['codes'].shape==(5,16,16,2)
            print(name,'synthetic fullcache parity+serialization PASS',flush=True)
    print('All synthetic checks passed')


if __name__=='__main__':main()
