#!/usr/bin/env python
"""Export the full four-expert digi16-nf model, not arbitrary symbolic formulas.

Cache: NPZ with n/X0/y0/... (original batches), or X/y (one batch).
X is analog charge shaped (N,16,16,2); y is already-normalized [x,y,cotA,cotB].
All events are used. --labels-scale declares the checkpoint's normalization; it
must match all four experts and any cache labels_scale metadata. No rescaling.
ADC defaults reproduce the original checkpoint construction; loaded checkpoint
thresholds are authoritative and recorded, and the ADC remains outside the RTL.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'two_bit_optimization_helpers'))
from symmoe_nf_ref import nll_v2

THRESHOLDS = [8.565593719482422, 20.227340698242188, 47.49168395996094]
LABELS_SCALE = [123.730140686, 30.942258835, 6.587896824, 1.848129153]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cache(path):
    with np.load(path, allow_pickle=False) as z:
        if 'n' in z:
            n = int(z['n']); assert n > 0
            xs = [z[f'X{i}'] for i in range(n)]
            ys = [z[f'y{i}'] for i in range(n)]
        else:
            xs, ys = [z['X']], [z['y']]
        for x, y in zip(xs, ys):
            assert x.ndim == 4 and x.shape[1:] == (16, 16, 2) and len(x) > 0, x.shape
            assert y.shape == (len(x), 4), y.shape
            assert np.isfinite(x).all() and np.isfinite(y).all()
        scale = z['labels_scale'].tolist() if 'labels_scale' in z else None
        return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
                np.array([len(y) for y in ys]), scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--validation-cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--thresholds', type=float, nargs=3, default=THRESHOLDS)
    ap.add_argument('--labels-scale', type=float, nargs=4, default=LABELS_SCALE)
    args = ap.parse_args()
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from models.symmoe_model import SymMoE_digi16nf_SoftQuantizer
    weights_hash, cache_hash = sha(args.weights), sha(args.validation_cache)
    X, labels, batches, cache_scale = load_cache(args.validation_cache)
    expected_scale = np.asarray(args.labels_scale)
    assert np.isfinite(expected_scale).all() and (expected_scale > 0).all()
    if cache_scale is not None:
        np.testing.assert_allclose(cache_scale, expected_scale, rtol=1e-6, atol=0)
    m = SymMoE_digi16nf_SoftQuantizer((16, 16, 2), args.thresholds, 0.0, np.arange(4, dtype=np.float32))
    m.load_weights(str(args.weights))  # strict topology load; never by_name or skip_mismatch
    for k in range(4):
        expert = m.get_layer(f'physics_ansatz_{k}')
        np.testing.assert_allclose(expert.labels_scale.numpy(), expected_scale, rtol=1e-6, atol=0)
        assert expert.variant == 'barycenter' and expert.width_mode == 'count'
        assert expert.support_min_code == 2 and expert.p_x == 50 and expert.p_y == 12.5 and expert.T == 100
    sq = m.get_layer('soft_quantizer_output')
    thresholds, levels = sq.thresholds.numpy(), sq.levels.numpy()
    np.testing.assert_array_equal(levels, [0, 1, 2, 3])
    assert np.isfinite(thresholds).all() and (np.diff(thresholds) > 0).all()
    codes = Model(m.input, sq.output).predict(X, batch_size=5000, verbose=0)
    assert np.isfinite(codes).all() and np.isin(codes, [0, 1, 2, 3]).all()
    pred = m.predict(X, batch_size=5000, verbose=0).astype(np.float32)
    cin = tf.keras.Input((16, 16, 2)); tensors = {sq.name: cin}
    for layer in m.layers[2:]:
        ins = layer.inbound_nodes[0].inbound_layers
        ins = ins if isinstance(ins, list) else [ins]
        a = [tensors[i.name] for i in ins]
        tensors[layer.name] = layer(a if len(a) > 1 else a[0])
    codes_model = Model(cin, tensors[m.layers[-1].name])
    pred_codes = codes_model.predict(codes, batch_size=5000, verbose=0)
    assert pred.shape == (len(X), 14) and np.isfinite(pred).all() and np.isfinite(pred_codes).all()
    np.testing.assert_array_equal(pred_codes, pred)
    W = {f'{layer.name}/' + w.name.split(':')[0]: w.numpy() for layer in m.layers for w in layer.weights}
    nll, per = nll_v2(labels, pred, batches)
    assert np.isfinite(nll) and sha(args.weights) == weights_hash and sha(args.validation_cache) == cache_hash
    metadata = dict(model='SymMoE_digi16nf', formula='four signed barycenter experts, count support >=2; router36->16->4; uncertainty32->8->10',
                    weights=str(args.weights.resolve()), weights_sha256=weights_hash,
                    validation_cache=str(args.validation_cache.resolve()), validation_cache_sha256=cache_hash,
                    n_val=len(X), batch_sizes=batches.tolist(), targets_already_normalized=True,
                    labels_scale=expected_scale.tolist(), input_shape=[16,16,2], target_order=['x','y','cotA','cotB'],
                    output_order=['x','d0','y','d1','cotA','d2','cotB','d3','M21','M31','M32','M41','M42','M43'],
                    geometry=dict(p_x_um=50,p_y_um=12.5,thickness_um=100), thresholds=thresholds.tolist(),
                    constructor_threshold_defaults=args.thresholds, levels=levels.tolist(), nll_float=nll,
                    nll_definition='equal mean of original validation batch sums; custom_loss_v2; not NLL/event',
                    codes_vs_analog_max_abs_error=0.0, export_script_sha256=sha(__file__),
                    model_source_sha256=sha(HERE.parents[2]/'two_bit_optimization_helpers/models/symmoe_model.py'))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/'digi16nf_export.npz', codes=codes.astype(np.uint8), labels=labels, keras=pred,
                        batch_sizes=batches, thresholds=thresholds, nll_float=nll, nll_float_batches=per,
                        metadata_json=json.dumps(metadata), **{f'w::{k}':v for k,v in W.items()})
    (args.out/'digi16nf_export.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
