"""Cache the O25 teacher's 14-dim predictions, and VERIFY its diagonal dialect.

Teacher: runs/o22_ep100/seed_22042 -- chosen because its readout (slices [10,21],
thresholds [8.57,20.23,47.49]) is IDENTICAL to what O25's students run frozen.
The project's best checkpoint (o24b_recover, -40,337) uses slices [10,18] and so
would have the teacher and student seeing different inputs; that is not a
distillation setup, it is two different experiments.

result.json records no diag_mode, and scoring a checkpoint under the wrong
dialect is pure artifact (conditional_nll.scale_tril), so we score under BOTH
and let the recorded best_val_loss = -40447.7 pick the winner.
"""
import os, sys, json, numpy as np, tensorflow as tf
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.abspath(os.path.join(_HERE,os.pardir,os.pardir))
sys.path.insert(0, os.path.join(_REPO,'two_bit_optimization_helpers'))
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.set_logical_device_configuration(g,[tf.config.LogicalDeviceConfiguration(memory_limit=6000)])
    except Exception: pass
from prepare_tfrecords import load_tfrecords
from train import create_model
from conditional_nll import nll_terms

THR=[8.565593719482422,20.227340698242188,47.49168395996094]
CKPT=os.path.join(_REPO,'runs/o22_ep100/seed_22042/best.weights.hdf5')
OUT=os.path.join(_REPO,'runs/o25_qconv2d/_teacher_cache'); os.makedirs(OUT,exist_ok=True)
BASE="/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
TFR=os.path.join(BASE,"TFR_files_all101_noise_contained_discovery")

m=create_model('ViT_MaxDeep_PairLattice', timeslices=101, soft_quantize_layer=True,
               initial_thresholds=THR, threshold_offset=0.0)
m.load_weights(CKPT)
print(f'teacher loaded: {m.count_params()} params', flush=True)

# exactly the driver's call, so the teacher sees the SAME batches the students do
tg,vg=load_tfrecords(os.path.join(TFR,'TFR_train'), os.path.join(TFR,'TFR_test'), noise=-1, seed=42)
for tag,gen in (('train',tg),('val',vg)):
    P=[];Y=[]
    for i in range(len(gen)):
        xb,yb=gen[i]
        P.append(np.asarray(m(xb,training=False),dtype=np.float32)); Y.append(np.asarray(yb,dtype=np.float32))
    # batches are ragged (last one is partial), so store per-batch like
    # PreSlicedCache does rather than stacking into one array
    with open(os.path.join(OUT,f'teacher_{tag}.npz'),'wb') as fh:
        np.savez(fh, n=len(P), **{f'P{i}':a for i,a in enumerate(P)},
                              **{f'Y{i}':a for i,a in enumerate(Y)})
    if tag=='val':
        for mode in ('softplus','relu'):
            n=sum(float(tf.reduce_sum(tf.reduce_sum(nll_terms(tf.cast(Y[b],tf.float64),
                    tf.cast(P[b],tf.float64),diag_mode=mode),axis=-1))) for b in range(len(P)))
            print(f'  val plain NLL under diag_mode={mode:9s}: {n/len(P):14.1f}  (recorded best_val_loss -40447.7)', flush=True)
    print(f'  cached {tag}: {len(P)} batches, shapes {P[0].shape} .. {P[-1].shape}', flush=True)
