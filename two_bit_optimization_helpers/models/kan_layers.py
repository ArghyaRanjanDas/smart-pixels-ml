"""Quantisation-aware KAN layers for the smart-pixel ASIC study (branch `kan`).

A KAN edge is a learned 1-D function phi(x); a KAN node SUMS its incoming edges.
Here every edge is a learnable TABLE over the quantised input levels, read by linear
interpolation. At the quantised input codes this is exactly the lookup table the
chip would store (KANELE-style, arXiv:2512.12850), so training and hardware agree
bit-for-bit; between codes the interpolation only supplies a gradient. Formally a
KAN with order-1 (linear) B-splines on a grid of `n_levels` points.

  QKANDense(units, n_levels, x_range, table_bits)
      inputs (..., d_in) on [x_range] taking n_levels values -> (..., units)
      d_in*units edges, each a table of n_levels entries quantised to table_bits
  QKANConv2D(filters, kernel_size, ...)
      the same edges slid over the image ('valid' padding, stride 1): one table set
      shared by every position, like a conv shares its weights. No multipliers.
"""
import tensorflow as tf
from qkeras import quantized_bits


class QKANDense(tf.keras.layers.Layer):
    def __init__(self, units, n_levels, x_range=(0.0, 3.0), table_bits=4, init_scale=1.0, **kw):
        super().__init__(**kw)
        self.units, self.n_levels = int(units), int(n_levels)
        self.lo, self.hi = float(x_range[0]), float(x_range[1])
        self.table_bits, self.init_scale = int(table_bits), float(init_scale)
        # same signed fixed-point grid QConv2D uses for its 4-bit weights
        self.q = quantized_bits(table_bits, 0, 1, alpha=1) if table_bits else None

    def build(self, input_shape):
        d_in = int(input_shape[-1]); self.d_in = d_in
        # init: each edge ~ a small LINEAR function w*x (w ~ N(0, 1/d_in)), i.e. the
        # layer starts out as an ordinary linear map, like a conv/dense init
        grid = tf.linspace(self.lo, self.hi, self.n_levels)
        centred = (grid - 0.5 * (self.lo + self.hi)) / (0.5 * (self.hi - self.lo) + 1e-9)   # [-1,1]
        w = tf.random.normal((d_in, self.units), stddev=self.init_scale / (d_in ** 0.5))
        init = w[:, :, None] * centred[None, None, :]                                         # (d_in,units,L)
        self.table = self.add_weight(name='table', shape=(d_in, self.units, self.n_levels),
                                     initializer=tf.keras.initializers.Constant(init.numpy()),
                                     trainable=True)

    def quantised_table(self):
        return self.q(self.table) if self.q is not None else self.table

    def call(self, x):
        # Linear interpolation between table entries == weighting the entries with
        # order-1 B-spline "hat" functions B_l(u) = max(0, 1 - |u - l|), i.e. the KAN
        # formula phi(x) = sum_l c_l B_l(x). One contraction instead of per-edge
        # gathers (identical values, much faster on a GPU).
        t = self.quantised_table()                                   # (d_in, units, L)
        L = self.n_levels
        u = (tf.clip_by_value(x, self.lo, self.hi) - self.lo) / (self.hi - self.lo) * (L - 1)
        basis = tf.nn.relu(1.0 - tf.abs(u[..., None] - tf.range(L, dtype=x.dtype)))   # (..., d_in, L)
        # as ONE 2-D matmul: (rows, d_in*L) @ (d_in*L, units). tf.einsum's backward
        # materialised a (units,units,L,batch,H,W) tensor (1.6 GB) and OOM'd.
        lead = tf.shape(basis)[:-2]
        b2 = tf.reshape(basis, (-1, self.d_in * L))
        t2 = tf.reshape(tf.transpose(t, (0, 2, 1)), (self.d_in * L, self.units))
        out = tf.matmul(b2, t2)                                      # node = SUM over edges d
        return tf.reshape(out, tf.concat([lead, [self.units]], 0))

    def stored_bits(self):
        return self.d_in * self.units * self.n_levels * (self.table_bits or 32)

    def get_config(self):
        c = super().get_config()
        c.update(units=self.units, n_levels=self.n_levels, x_range=(self.lo, self.hi),
                 table_bits=self.table_bits, init_scale=self.init_scale)
        return c


class QKANConv2D(tf.keras.layers.Layer):
    """KAN convolution: QKANDense applied to every kxk patch (valid, stride 1)."""
    def __init__(self, filters, kernel_size, n_levels, x_range=(0.0, 3.0), table_bits=4, **kw):
        super().__init__(**kw)
        self.filters, self.k = int(filters), int(kernel_size)
        self.dense = QKANDense(filters, n_levels, x_range, table_bits, name=kw.get('name', 'kanconv') + '_edges')

    def call(self, x):
        if self.k == 1:
            return self.dense(x)
        p = tf.image.extract_patches(x, sizes=[1, self.k, self.k, 1], strides=[1, 1, 1, 1],
                                     rates=[1, 1, 1, 1], padding='VALID')   # (B,H',W',k*k*C)
        return self.dense(p)

    def stored_bits(self):
        return self.dense.stored_bits()
