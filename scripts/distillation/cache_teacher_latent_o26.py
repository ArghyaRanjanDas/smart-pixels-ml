"""O26 (JEPA-lite): cache the frozen teacher's 64-dim penultimate latent (dense_11).

Same teacher, same generator call and batch order as cache_teacher_o25.py, so the
latent arm and the existing KL arms are distilled from the identical model on the
identical events. Saves per-batch arrays (batches are ragged) plus TRAIN-set
standardisation stats; dead ReLU units (never > 0 on train) are flagged so the
student is not asked to regress a constant.

Checks built in, against the O25 KL cache:
  * labels EXACTLY equal batch by batch -> the same events in the same order;
  * teacher outputs agree to 1e-3 relative -> the same checkpoint. Not bit-exact:
    a fresh process picks different GPU kernels, and the 711K-param teacher then
    differs at ~4e-4 relative (measured; identical with TF32 on or off, and
    exactly reproducible within one process). A wrong checkpoint or batch order
    would differ at O(1).
"""
import os, sys, json, numpy as np, tensorflow as tf
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_R, 'two_bit_optimization_helpers'))
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=6000)])
    except Exception: pass
from prepare_tfrecords import load_tfrecords
from train import create_model

THR = [8.565593719482422, 20.227340698242188, 47.49168395996094]
CKPT = os.path.join(_R, 'runs/o22_ep100/seed_22042/best.weights.hdf5')
KLC = os.path.join(_R, 'runs/o25_qconv2d/_teacher_cache')
OUT = os.path.join(_R, 'runs/o26_jepa/_teacher_latent'); os.makedirs(OUT, exist_ok=True)
TFR = ("/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_"
       "convolved_to_200ps/shuffled_3d/TFR_files_all101_noise_contained_discovery")

m = create_model('ViT_MaxDeep_PairLattice', timeslices=101, soft_quantize_layer=True,
                 initial_thresholds=THR, threshold_offset=0.0)
m.load_weights(CKPT)
two = tf.keras.Model(m.inputs, [m.get_layer('dense_11').output, m.outputs[0]])
tg, vg = load_tfrecords(os.path.join(TFR, 'TFR_train'), os.path.join(TFR, 'TFR_test'), noise=-1, seed=42)

lat = {}
for tag, gen in (('train', tg), ('val', vg)):
    ref = np.load(os.path.join(KLC, f'teacher_{tag}.npz'))
    Z, worst = [], 0.0
    assert int(ref['n']) == len(gen), f'{tag}: {len(gen)} batches vs {int(ref["n"])} cached'
    for i in range(len(gen)):
        xb, yb = gen[i]
        assert np.array_equal(np.asarray(yb, np.float32), ref[f'Y{i}']), f'{tag} batch {i}: different events'
        z, p = two(xb, training=False)
        worst = max(worst, float(np.max(np.abs(np.asarray(p) - ref[f'P{i}'])) / np.max(np.abs(ref[f'P{i}']))))
        Z.append(np.asarray(z, np.float32))
    assert worst < 1e-3, f'{tag}: teacher output differs from the O25 KL cache by {worst:.2e} relative'
    print(f'{tag}: {len(Z)} batches, labels identical, teacher output within {worst:.1e} relative of the KL cache', flush=True)
    lat[tag] = Z

allz = np.concatenate(lat['train'], 0)
mu, sd = allz.mean(0), allz.std(0)
dead = (allz.max(0) <= 0)
sd_safe = np.where(sd > 1e-6, sd, 1.0)
for tag, Z in lat.items():
    with open(os.path.join(OUT, f'latent_{tag}.npz'), 'wb') as fh:
        np.savez(fh, n=len(Z), **{f'Z{i}': ((z - mu) / sd_safe).astype(np.float32) for i, z in enumerate(Z)})
json.dump({'layer': 'dense_11', 'dim': int(allz.shape[1]), 'n_dead': int(dead.sum()),
           'dead_units': np.where(dead)[0].tolist(), 'mean': mu.tolist(), 'std': sd.tolist(),
           'teacher': CKPT}, open(os.path.join(OUT, 'stats.json'), 'w'), indent=1)
print(f'latent dim {allz.shape[1]}, dead ReLU units {int(dead.sum())}, '
      f'live-unit std range {sd[~dead].min():.3g}..{sd[~dead].max():.3g}', flush=True)
