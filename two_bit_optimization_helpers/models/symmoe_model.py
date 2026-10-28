"""digi16 (symbolic MoE, slim16 + 2-bit input) with the full four-expert signed barycenter formula.

The functional Keras model preserves checkpoint topology so that
`.layers`, `get_layer('soft_quantizer_output')`, weight save/resume and the driver's threshold
freezing preserve the original checkpoint interface.

  input (16,16,2) -> SoftQuantizeLayer (2-bit codes 0..3, name 'soft_quantizer_output')
    -> RouterFeatures (36) -> Dense16 relu -> Dense4 -> softmax          (router, slim16)
    -> 4 x PhysicsAnsatz(barycenter, signed v3)  (ansatz_signed.py)       (experts, no weights but ~12 scalars)
    -> mix means with the router weights
    -> shared build_error_mlp (10 Cholesky entries)
    -> pack_14  ->  14 outputs [x,d0,y,d1,cotA,d2,cotB,d3, 6 off-diagonals]

digi16 and digi16pc are ONE arm here (thresholds are frozen at the study's and the layer is hard in
the forward pass, k only shapes the backward pass).  'SymMoE_digi16nf' = same with the code-sum>=2
support mask in the ansatz widths/drift signs.

LABELS_SCALE gives the original dataset normalization, not an arbitrary dataset default.
Portable hardware export strictly loads the complete checkpoint and checks every expert's
normalization. Optional SYMMOE_SIGN_INIT is used only for fresh model initialization;
no local training-cache or run directory is discovered automatically.
"""
import json
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model

from SoftQuantizeLayer import SoftQuantizeLayer
from models.error_mlp import build_error_mlp
from models.student_max import pack_14
from symbolic.ansatz_signed import PhysicsAnsatz

LABELS_SCALE = [123.730140686, 30.942258835, 6.587896824, 1.848129153]  # this repo's 2_5 train metadata
N_EXPERTS = 4

_KEYMAP = {'k_alpha': 'initial_sign_k_alpha', 'a0_alpha': 'initial_sign_a0_alpha',
           'k_beta': 'initial_sign_k_beta', 'a0_beta': 'initial_sign_a0_beta'}


def _sign_init(variant):
    # Portable export loads all trained scalars strictly from --weights.
    # Optional initialization is only for callers constructing a fresh model.
    path = os.environ.get('SYMMOE_SIGN_INIT')
    if not path:
        return {}
    d = json.load(open(path))
    return {_KEYMAP.get(k, k): float(v) for k, v in d.items()}


class RouterFeatures(Layer):  # verbatim from Shiqi's notebook cell 5 / symmoe_slim_train.py
    def call(self, x):
        xp = tf.nn.relu(x)
        q = tf.reduce_sum(xp, axis=-1)
        px = tf.reduce_sum(q, axis=2)
        py = tf.reduce_sum(q, axis=1)
        tot = tf.reduce_sum(py, axis=1, keepdims=True) + 1e-9
        pxn, pyn = px / tot, py / tot
        yc = tf.range(16, dtype=tf.float32) - 7.5
        mu = tf.reduce_sum(pyn * yc, axis=1, keepdims=True)
        d = yc[None, :] - mu
        m2 = tf.reduce_sum(pyn * d**2, axis=1, keepdims=True)
        m3 = tf.reduce_sum(pyn * d**3, axis=1, keepdims=True)
        std = tf.sqrt(m2 + 1.0)
        skew = m3 / (std**3)
        py0 = tf.reduce_sum(xp[..., 0], axis=1)
        py1 = tf.reduce_sum(xp[..., -1], axis=1)
        c0 = tf.reduce_sum(py0 * yc, 1, keepdims=True) / (tf.reduce_sum(py0, 1, keepdims=True) + 1e-9)
        c1 = tf.reduce_sum(py1 * yc, 1, keepdims=True) / (tf.reduce_sum(py1, 1, keepdims=True) + 1e-9)
        tasym = c1 - c0
        logq = tf.math.log(tot + 1.0)
        f = tf.concat([pxn, pyn, std, skew, tasym, logq], axis=-1)
        f = tf.where(tf.math.is_finite(f), f, tf.zeros_like(f))
        return tf.clip_by_value(f, -8.0, 8.0)


class MixMeans(Layer):
    """sum_k w[:,k] * mu_k  (router weights sum to 1, so no Cholesky mixing)."""
    def call(self, inputs):
        w, *mus = inputs
        return tf.add_n([w[:, k:k + 1] * mus[k] for k in range(len(mus))])


def _build(shape, initial_thresholds, threshold_offset, initial_levels, variant, support_min_code,
           trainable_thresholds=True):
    x_in = Input(shape)
    xq = SoftQuantizeLayer(
        n_bits=2, initial_thresholds=initial_thresholds, threshold_offset=threshold_offset,
        initial_levels=initial_levels, trainable_levels=False,
        trainable_thresholds=trainable_thresholds, initial_k=1.0, trainable_k=True,
        name='soft_quantizer_output')(x_in)
    f = RouterFeatures(name='router_features')(xq)
    h = Dense(16, activation='relu', name='router_hidden')(f)
    w = tf.keras.layers.Softmax(axis=-1, name='router_softmax')(Dense(N_EXPERTS, name='router_out')(h))
    si = _sign_init(variant)
    mus = [PhysicsAnsatz(variant='barycenter', labels_scale=LABELS_SCALE, width_mode='count',
                         support_min_code=support_min_code, name=f'physics_ansatz_{k}', **si)(xq)
           for k in range(N_EXPERTS)]
    means = MixMeans(name='mix_means')([w] + mus)
    chol = build_error_mlp(shape=tuple(shape), name='error_mlp')(xq)
    out = tf.keras.layers.Lambda(lambda t: pack_14(t[0], t[1]), name='pack_14')([means, chol])
    return Model(x_in, out, name=f'symmoe_{variant}')


def SymMoE_digi16_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None,
                                trainable_thresholds=True):
    return _build(shape, initial_thresholds, threshold_offset, initial_levels, 'digi16', 1.0,
                  trainable_thresholds)


def SymMoE_digi16nf_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None,
                                  trainable_thresholds=True):
    return _build(shape, initial_thresholds, threshold_offset, initial_levels, 'digi16nf', 2.0,
                  trainable_thresholds)
