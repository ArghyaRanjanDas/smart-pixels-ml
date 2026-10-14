"""Export the full 101-slice noise-contained dataset to .npy for the PyTorch V-JEPA runs.

Exactly what load_tfrecords(noise=-1, seed=42) yields to every TF model in this
project -- same events, same order, same preprocessing -- so JEPA results are
comparable with the ledger. x: float16 (N,16,16,101), y: float32 (N,4) scaled labels.
Float16 is checked against float32 on every batch (max abs/rel error printed).
"""
import os, sys, numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(R, 'two_bit_optimization_helpers'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from prepare_tfrecords import load_tfrecords
TFR = ("/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_"
       "convolved_to_200ps/shuffled_3d/TFR_files_all101_noise_contained_discovery")
OUT = os.path.join(R, 'runs/o26_jepa/data')
tg, vg = load_tfrecords(os.path.join(TFR, 'TFR_train'), os.path.join(TFR, 'TFR_test'), noise=-1, seed=42)
for tag, gen in (('train', tg), ('val', vg)):
    n = sum(np.asarray(gen[i][1]).shape[0] for i in range(len(gen)))
    X = np.lib.format.open_memmap(os.path.join(OUT, f'{tag}_x.npy'), mode='w+', dtype=np.float16, shape=(n, 16, 16, 101))
    Y = np.zeros((n, 4), np.float32); k = 0; ea = er = 0.0; lo, hi = np.inf, -np.inf
    for i in range(len(gen)):
        xb, yb = gen[i]; xb = np.asarray(xb, np.float32); m = xb.shape[0]
        h = xb.astype(np.float16)
        ea = max(ea, float(np.max(np.abs(h.astype(np.float32) - xb))))
        er = max(er, float(np.max(np.abs(h.astype(np.float32) - xb) / (np.abs(xb) + 1e-3))))
        lo, hi = min(lo, float(xb.min())), max(hi, float(xb.max()))
        X[k:k + m] = h; Y[k:k + m] = np.asarray(yb, np.float32); k += m
    X.flush(); np.save(os.path.join(OUT, f'{tag}_y.npy'), Y)
    print(f'{tag}: {n} events, x range [{lo:.3f}, {hi:.3f}], float16 max abs err {ea:.2e}, max rel err {er:.2e}', flush=True)
