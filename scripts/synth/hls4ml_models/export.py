"""Export baseline checkpoint + complete preslice validation cache to a codes model.

The digitizer stays outside the hardware. No training or synthesis is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=['QMlp_Max', 'QConv2D_Max'])
    ap.add_argument('--weights', required=True, type=Path)
    ap.add_argument('--validation-cache', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--thresholds', type=float, nargs=3,
                    default=[8.565593719482422, 20.227340698242188, 47.49168395996094],
                    help='Initializer only; checkpoint thresholds replace these')
    ap.add_argument('--batch-size', type=int, default=1000)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh export directory: '+str(args.out))
    if args.batch_size <= 0:
        raise ValueError('batch-size must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_USE_LEGACY_KERAS'] = '0'  # QKeras uses standalone Keras2, including subprocesses
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[key] = '1'
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / 'two_bit_optimization_helpers'))
    import numpy as np
    import tensorflow as tf
    from qkeras.utils import _add_supported_quantized_objects
    from models import mlp_encoder_model_quantized as mlp
    from models import conv2d_model_quantized as conv
    tf.keras.backend.clear_session()
    levels = np.arange(4, dtype=np.float32)
    if args.model == 'QMlp_Max':
        full = mlp.QMlp_Max_SoftQuantizer((16, 16, 2), args.thresholds, 0., levels)
    else:
        full = conv.QConv2D_Max_SoftQuantizer((16, 16, 2), 5, 3, args.thresholds, 0., levels)
    print('model graph', args.model, type(full).__module__, len(full.layers), len(full.weights), flush=True)
    checkpoint_hash = sha(args.weights)
    full.load_weights(str(args.weights))
    quantizer = full.get_layer('soft_quantizer_output')
    encoder = tf.keras.Model(full.input, quantizer.output)
    # Explicit codes input keeps the validated converter/testbench interface stable.
    cin = tf.keras.Input((16, 16, 2), name='codes')
    if args.model == 'QMlp_Max':
        out = mlp._var_network(mlp._mlp_encoder_network(cin), hidden=16, output=14)
    else:
        out = conv._conv_network(cin)
        out = tf.keras.layers.AveragePooling2D(pool_size=(3, 3))(out)
        from qkeras import QActivation
        out = QActivation('quantized_bits(8, 0, alpha=1)')(out)
        out = conv._var_network(out, hidden=16, output=14)
    codes_model = tf.keras.Model(cin, out, name=args.model+'_codes')
    source = [layer for layer in full.layers if layer.weights and layer is not quantizer]
    target = [layer for layer in codes_model.layers if layer.weights]
    if len(source) != len(target):
        raise ValueError('Weighted-layer count mismatch')
    for old, new in zip(source, target):
        if [w.shape for w in old.weights] != [w.shape for w in new.weights]:
            raise ValueError(f'Weight shape mismatch: {old.name}, {new.name}')
        new.set_weights(old.get_weights())
    outputs, codes, labels, sizes = [], [], [], []
    with np.load(args.validation_cache, allow_pickle=False) as cache:
        count = int(cache['n'])
        if count <= 0:
            raise ValueError('Empty cache')
        for i in range(count):
            x, y = np.asarray(cache[f'X{i}'], np.float32), np.asarray(cache[f'y{i}'], np.float32)
            if x.shape != (len(y), 16, 16, 2) or y.shape != (len(y), 4) or not len(y):
                raise ValueError(f'Invalid original cache batch {i}')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite cache values')
            batch_c, batch_p = [], []
            for j in range(0, len(x), args.batch_size):
                raw = x[j:j+args.batch_size]
                c = encoder(raw, training=False).numpy()
                if not np.isin(c, levels).all():
                    raise ValueError('Digitizer did not emit codes0..3')
                p = full(raw, training=False).numpy()
                stripped = codes_model(c, training=False).numpy()
                np.testing.assert_array_equal(p, stripped)
                batch_c.append(c.astype(np.uint8)); batch_p.append(p)
            codes.append(np.concatenate(batch_c)); outputs.append(np.concatenate(batch_p))
            labels.append(y); sizes.append(len(y))
            print(f'full validation batch {i+1}/{count}: {len(y)} events', flush=True)
    x, y, p = np.concatenate(codes), np.concatenate(labels), np.concatenate(outputs)
    if p.shape != (len(y), 14) or not np.isfinite(p).all():
        raise ValueError('Expected finite14-output predictions')
    if sha(args.weights) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during export')
    args.out.mkdir(parents=True)
    h5 = args.out / 'codes_model.h5'
    codes_model.save(str(h5), include_optimizer=False)
    co = {}; _add_supported_quantized_objects(co)
    reloaded = tf.keras.models.load_model(h5, custom_objects=co, compile=False)
    for j in range(0, len(x), args.batch_size):
        np.testing.assert_array_equal(reloaded(x[j:j+args.batch_size], training=False).numpy(), p[j:j+args.batch_size])
    manifest = dict(model=args.model, outputs=14, input_shape=[16,16,2],
                    output_order=['x','raw_L11','y','raw_L22','cotA','raw_L33','cotB','raw_L44',
                                  'L21','L31','L32','L41','L42','L43'],
                    checkpoint=dict(path=str(args.weights.resolve()), sha256=checkpoint_hash,
                                    selection_scope='user-supplied weights; epoch/selection not inferred'),
                    validation_cache=dict(path=str(args.validation_cache.resolve()), sha256=sha(args.validation_cache)),
                    validation_events=len(y), original_batch_sizes=sizes, all_cache_events=True,
                    full_vs_codes_only_exact_events=len(y), saved_reload_exact_events=len(y),
                    thresholds=quantizer.thresholds.numpy().tolist(), levels=quantizer.levels.numpy().tolist(),
                    software_parameters=full.count_params(), hardware_coefficients=codes_model.count_params(),
                    tensorflow_version=tf.__version__, model_file=h5.name, model_sha256=sha(h5),
                    source_sha256={str(Path(mod.__file__).relative_to(repo)):sha(mod.__file__) for mod in (mlp,conv)})
    np.savez_compressed(args.out/'bundle.npz', codes=x, keras=p, labels=y,
                        batch_sizes=np.asarray(sizes), thresholds=quantizer.thresholds.numpy(),
                        provenance_json=json.dumps(manifest))
    manifest['bundle_sha256'] = sha(args.out/'bundle.npz')
    (args.out/'export_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
