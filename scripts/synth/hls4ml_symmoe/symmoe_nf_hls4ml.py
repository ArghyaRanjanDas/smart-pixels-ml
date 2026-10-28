"""hls4ml 1.1.0 extension for digi16-nf (SymMoE_digi16nf): the whole post-digitiser model as ONE hls4ml layer.

Extension components:
  Keras layer SymMoENFHW  (holds the 21 parameter ports, already on their ap_fixed grids; its call() is a stub,
                           the reference is symmoe_nf_ref.fixed_ref)
  register_keras_layer_handler('SymMoENFHW', parse)     Keras -> IR dict
  register_layer('SymMoENF', SymMoENF)                  IR layer: 21 weight variables, output ap_fixed<IO+FO, IO>
  Vitis config/function templates + nnet_symmoe_nf.h   (the bit-exact kernel)

    import symmoe_nf_hls4ml as S; S.register('Vitis'); m = S.build_keras_model(Q, fm, cfg)

Programmable build: Model BramFactor=0 -> every parameter array becomes a top-level port (as the other *_prog designs).
Fixed build: BramFactor large -> the parameters are constants in firmware/weights/*.h and Vitis folds them.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import symmoe_nf_ref as R          # noqa: E402
import symmoe_nf_hw as H           # noqa: E402

import hls4ml                                                     # noqa: E402
from hls4ml.model.attributes import Attribute, TypeAttribute, WeightAttribute   # noqa: E402
from hls4ml.model.types import FixedPrecisionType                 # noqa: E402
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate  # noqa: E402

KERNEL_H = HERE / 'nnet_symmoe_nf.h'
PNAMES = [p for p, _, _ in R.PORTS]

# ------------------------------------------------------------------ Keras side
import tensorflow as tf            # noqa: E402


class SymMoENFHW(tf.keras.layers.Layer):
    """codes (16,16,2) -> 14 outputs. Parameter container for the converter; call() returns zeros."""

    def __init__(self, hw=None, **kw):
        super().__init__(**kw)
        self.hw = hw if isinstance(hw, str) else json.dumps(hw)

    def build(self, input_shape):
        for p, shape, _ in R.PORTS:
            setattr(self, p, self.add_weight(name=p, shape=shape, initializer='zeros', trainable=False,
                                             dtype='float64'))
        super().build(input_shape)

    def call(self, x):
        return tf.zeros_like(tf.reshape(x, (-1, 512))[:, :14])

    def get_config(self):
        c = super().get_config(); c['hw'] = self.hw
        return c


def build_keras_model(Q, fm, cfg, name='symmoe_nf'):
    inp = tf.keras.Input((16, 16, 2), name='codes')
    L = SymMoENFHW(hw=dict(cfg=cfg, formats={k: list(v) for k, v in fm.items()}), name=name)
    m = tf.keras.Model(inp, L(inp), name='digi16nf_hw')
    PF = H.ports_float(Q, fm)
    L.set_weights([PF[p].reshape(s) for p, s, _ in R.PORTS])
    return m


# ------------------------------------------------------------------ parser / IR
def parse_symmoe_nf(keras_layer, input_names, input_shapes, data_reader):
    cfg = keras_layer['config']
    assert keras_layer['class_name'] == 'SymMoENFHW'
    assert tuple(input_shapes[0][1:]) == (16, 16, 2), input_shapes
    layer = {'class_name': 'SymMoENF', 'name': cfg['name'], 'hw': cfg['hw'], 'n_in': 512, 'n_out': 14}
    for p in PNAMES:
        d = data_reader.get_weights_data(cfg['name'], p)
        assert d is not None, p
        layer[p + '_data'] = np.asarray(d, np.float64)
    if input_names is not None:
        layer['inputs'] = input_names
    return layer, [input_shapes[0][0], 14]


class SymMoENF(hls4ml.model.layers.Layer):
    _expected_attributes = [Attribute('n_in'), Attribute('n_out'), Attribute('hw', value_type=str)] + \
        [WeightAttribute(p) for p in PNAMES] + [TypeAttribute(p) for p in PNAMES]

    def initialize(self):
        hw = json.loads(self.get_attr('hw'))
        cfg, fm = hw['cfg'], {k: tuple(v) for k, v in hw['formats'].items()}
        H.check_cfg(cfg)
        for p, shape, _ in R.PORTS:
            Wd, I = fm[p]
            d = np.asarray(self.get_attr(p + '_data'), np.float64).reshape(-1)
            assert d.size == int(np.prod(shape)), (p, d.shape, shape)
            raw = d * 2.0 ** (Wd - I)
            assert np.array_equal(raw, np.rint(raw)) and raw.min() >= -(1 << (Wd - 1)) and raw.max() < (1 << (Wd - 1)), \
                f'{p}: values off the ap_fixed<{Wd},{I}> grid'
            self.add_weights_variable(name=p, var_name=f's{{index}}_{p}', type_name=f's{{index}}_{p}_t',
                                      precision=FixedPrecisionType(width=Wd, integer=I, signed=True), data=d)
        self.set_attr('cfg_dict', cfg)
        WO = cfg['IO'] + cfg['FO']
        self.add_output_variable([14], [f'N_OUT_{self.index}'],
                                 precision=FixedPrecisionType(width=WO, integer=cfg['IO'], signed=True))


class SymMoENFConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SymMoENF)

    def format(self, node):
        cfg = node.get_attr('cfg_dict')
        tn = {p: node.get_weights(p).type.name for p in PNAMES}
        return H.cpp_tables(cfg, f'symnf{node.index}') + H.cpp_config(cfg, tn, f'config{node.index}')


class SymMoENFFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SymMoENF, include_header=['nnet_utils/nnet_symmoe_nf.h'])

    def format(self, node):
        p = self._default_function_params(node)
        w = ', '.join(node.get_weights(n).name for n in PNAMES)
        i = node.index
        return (f'nnet::symmoe_nf<{p["input_t"]}, {p["output_t"]}, {p["config"]}>({p["input"]}, {p["output"]}, {w}, '
                f'symnf{i}_ln, symnf{i}_exp2, symnf{i}_tanh);')


_registered = set()


def register(backend_id='Vitis'):
    if 'SymMoENFHW' not in hls4ml.converters.get_supported_keras_layers():
        hls4ml.converters.register_keras_layer_handler('SymMoENFHW', parse_symmoe_nf)
    if 'SymMoENF' not in hls4ml.model.layers.layer_map:
        hls4ml.model.layers.register_layer('SymMoENF', SymMoENF)
    if backend_id not in _registered:
        backend = hls4ml.backends.get_backend(backend_id)
        backend.register_template(SymMoENFConfigTemplate)
        backend.register_template(SymMoENFFunctionTemplate)
        backend.register_source(KERNEL_H)
        _registered.add(backend_id)
