"""Three-arm OVERLAY of the O25 summary and pull plots, one representative seed each.

Imports make_perf_plots_o25 rather than copying it, so predict_df, the softplus
diagonal, SCALE and inverse_cot are literally the same code that produced the
per-seed slides -- an overlay that disagreed with its own single-seed panels
would be worse than no overlay.

The three arms share ONE validation set, so the true-value axis is identical and
the 15 bins line up exactly; only the predictions differ. That is what makes the
overlay honest.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaborn as sns
from scipy.optimize import curve_fit

import make_perf_plots_o25 as P
SCALE, inverse_cot, gauss = P.SCALE, P.inverse_cot, P.gauss
pi = np.pi

OUT = os.environ.get('SMARTPIX_OUT', P.OUT)
os.makedirs(OUT, exist_ok=True)

# (label, run root, kind, n_filters, seed, colour)
ARMS = [
    ("quant + constrained",   "runs/o25_qconv2d", "q", 5, "50842", "#7c3aed"),
    ("quant + unconstrained", "runs/o25_plain",   "q", 5, "50242", "#0ea5e9"),
    ("float + constrained",   "runs/o25_float",   "f", 5, "50342", "#e07b00"),
]

def ckpt(root, sd):
    return os.path.join(P.R, root, f's{sd}', f'seed_{sd}', 'best.weights.hdf5')

# --- overlay variants of resid_pos / resid_ang -----------------------------
# Same maths as the single-seed versions; the only changes are an explicit
# colour, a shared binning taken from the FIRST arm so every arm's points sit on
# the same 15 bin centres, and no per-call axis relabelling.

def _band(ax, xc, res, sig_up, sig_dn, edges, nb, color):
    up, dn = [], []
    for i in range(nb):
        m = (xc > edges[i]) & (xc < edges[i] + (edges[1] - edges[0]))
        mu = np.mean(res[m]) if m.any() else 0.0
        up.append(mu + (np.mean(sig_up[m]) if m.any() else 0.0))
        dn.append(mu - (np.mean(sig_dn[m]) if m.any() else 0.0))
    # Three FILLED bands stack into grey mud, so the band is drawn as its two
    # edges with only a whisper of fill -- each arm's predicted sigma stays
    # separable while the points keep the reader's eye.
    xs = np.linspace(edges[0], edges[0] + nb * (edges[1] - edges[0]), nb)
    ax.fill_between(xs, up, dn, alpha=0.05, color=color, lw=0, zorder=1)
    ax.plot(xs, up, color=color, lw=1.0, ls=(0, (4, 2)), alpha=.75, zorder=2)
    ax.plot(xs, dn, color=color, lw=1.0, ls=(0, (4, 2)), alpha=.75, zorder=2)

def resid_pos_ov(ax, df, v1, v2, name, scaling, color, lim):
    a = df[v1].values * scaling; b = df[v2].values * scaling; res = a - b
    nb = 15; xmin, xmax = lim; step = (xmax - xmin) / nb
    sns.regplot(x=a, y=res, x_bins=np.linspace(xmin, xmax, nb), fit_reg=None,
                marker='.', ax=ax, color=color,
                scatter_kws={'s': 26, 'zorder': 5}, line_kws={'lw': 0})
    ax.set_xlabel('True ' + name); ax.set_ylabel('True - pred ' + name)
    sig = df['sigma' + v2].values * scaling
    _band(ax, a, res, sig, sig, [xmin + i * step for i in range(nb + 1)], nb, color)
    return float(np.std(res))

def resid_ang_ov(ax, df, v1, v2, name, scaling, color, lim):
    ang  = inverse_cot(df[v2].values * scaling) * 180 / pi
    angT = inverse_cot(df[v1].values * scaling) * 180 / pi
    aup = np.abs(inverse_cot((df[v2].values + df['sigma' + v2].values) * scaling) * 180 / pi - ang)
    adn = np.abs(inverse_cot((df[v2].values - df['sigma' + v2].values) * scaling) * 180 / pi - ang)
    nb = 15; xmin, xmax = lim; step = (xmax - xmin) / nb
    res = angT - ang
    sns.regplot(x=angT, y=res, x_bins=np.linspace(xmin, xmax, nb), fit_reg=None,
                marker='.', ax=ax, color=color,
                scatter_kws={'s': 26, 'zorder': 5}, line_kws={'lw': 0})
    ax.set_xlabel('True ' + name); ax.set_ylabel('True - pred ' + name)
    _band(ax, angT, res, aup, adn, [xmin + i * step for i in range(nb + 1)], nb, color)
    return float(np.std(res))

def pull_ax_ov(ax, df, var, name, color):
    h = ax.hist(df['pull' + var], bins=np.linspace(-5, 5, 50), histtype='step',
                color=color, lw=1.3)
    _e = h[1]; xd = _e[:-1] + (_e[1] - _e[0]) / 2.0; yd = h[0]
    try:
        p, _ = curve_fit(gauss, xd, yd, p0=[max(yd.max(), 1), 0, 1], maxfev=10000)
        mu, sg = p[1], abs(p[2])
        xb = np.linspace(-5, 5, 200)
        ax.plot(xb, gauss(xb, *p), color=color, lw=1.0, ls='--', alpha=.85)
    except Exception:
        mu, sg = float(df['pull' + var].mean()), float(df['pull' + var].std())
    ax.set_xlabel(name); ax.set_yscale('log')
    return mu, sg, float(yd.max())

# --------------------------------------------------------------------------
SETS = {
  # the complete 2x2 of the campaign design
  'overlay4': ("Four-arm overlay - the full constraints x quantisation 2x2", [
      ("quant + constrained",   "runs/o25_qconv2d",    "q", 5, "50842", "#7c3aed"),
      ("quant + unconstrained", "runs/o25_plain",      "q", 5, "50242", "#0ea5e9"),
      ("float + constrained",   "runs/o25_float",      "f", 5, "50342", "#e07b00"),
      ("float + unconstrained", "runs/o25_float_plain","f", 5, "50842", "#15803d"),
  ]),
  # the three seeds the user's original pick used
  'overlay3': ("Three-arm overlay - one representative seed each", ARMS),
  # why the 4th cell has no single representative seed. NB the seeds differ in
  # angular WIDTH, not in bias -- x and y lie on top of each other and all three
  # carry the same large regional sweep. Labelling s50842 "good" (an earlier
  # draft did) reads its better NLL as a better model; it is a narrower one.
  'bimodal': ("float + unconstrained - all three seeds: same bias, different angular width", [
      ("s50842  -25,748  " + u"\u03c3\u03b1 21.6\u00b0", "runs/o25_float_plain", "f", 5, "50842", "#15803d"),
      ("s50242  -19,212  " + u"\u03c3\u03b1 63.2\u00b0", "runs/o25_float_plain", "f", 5, "50242", "#dc2626"),
      ("s50342  -19,334  " + u"\u03c3\u03b1 58.3\u00b0", "runs/o25_float_plain", "f", 5, "50342", "#b45309"),
  ]),
}

_PRED = {}
def pred(root, kind, nf, sd):
    k = (root, kind, nf, sd)
    if k not in _PRED:
        _PRED[k] = P.predict_df(ckpt(root, sd), kind, nf)
    return _PRED[k]


def render(name, suptitle, arms, legend_nll=True):
    loaded = [(l, sd, col) + pred(root, kind, nf, sd) for l, root, kind, nf, sd, col in arms]
    d0 = loaded[0][3]
    for e in loaded[1:]:
        assert np.allclose(d0['xtrue'], e[3]['xtrue']), 'arms disagree on the validation set'
    LIM = {
        'x':    (float((d0['xtrue'] * SCALE[0]).min()), float((d0['xtrue'] * SCALE[0]).max())),
        'y':    (float((d0['ytrue'] * SCALE[1]).min()), float((d0['ytrue'] * SCALE[1]).max())),
        'cotA': tuple(np.percentile(inverse_cot(d0['cotAtrue'].values * SCALE[2]) * 180 / pi, [0, 100])),
        'cotB': tuple(np.percentile(inverse_cot(d0['cotBtrue'].values * SCALE[3]) * 180 / pi, [0, 100])),
    }
    # ---- summary ----
    fig, ax = plt.subplots(2, 2, figsize=(9, 7)); fig.tight_layout(pad=4.0)
    res = {}
    for label, sd, col, df, nll in loaded:
        res[label] = {
          'x':    resid_pos_ov(ax[0][0], df, 'xtrue', 'x', r'$x$ [$\mu$m]', SCALE[0], col, LIM['x']),
          'y':    resid_pos_ov(ax[0][1], df, 'ytrue', 'y', r'$y$ [$\mu$m]', SCALE[1], col, LIM['y']),
          'cotA': resid_ang_ov(ax[1][0], df, 'cotAtrue', 'cotA', r'$\alpha$ [deg]', SCALE[2], col, LIM['cotA']),
          'cotB': resid_ang_ov(ax[1][1], df, 'cotBtrue', 'cotB', r'$\beta$ [deg]', SCALE[3], col, LIM['cotB'])}
    for a in ax.ravel():
        a.axhline(0, ls='--', lw=1.0, c='#334155', zorder=0)
    fig.suptitle(suptitle, fontsize=11, y=1.0)
    ncol = 2 if len(loaded) == 4 else len(loaded)
    fig.legend(handles=[Line2D([], [], color=c, marker='o', ls='none', ms=6,
                               label=(f"{l}  \u00b7  s{s}  \u00b7  {n:,.0f}" if legend_nll else l))
                        for l, s, c, _, n in loaded],
               loc='lower center', ncol=ncol, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.09 if ncol == 2 else -0.035))
    sp = os.path.join(OUT, f'{name}_summary.png')
    fig.savefig(sp, dpi=140, bbox_inches='tight', facecolor='white'); plt.close(fig)

    # ---- pull ----
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 6))
    stats, ymax = {}, {}
    for label, sd, col, df, nll in loaded:
        st = {}
        for a, v, nm in ((ax[0][0], 'x', r'$x$ pull'), (ax[0][1], 'y', r'$y$ pull'),
                         (ax[1][0], 'cotA', r'$\cot\alpha$ pull'), (ax[1][1], 'cotB', r'$\cot\beta$ pull')):
            mu, sg, ym = pull_ax_ov(a, df, v, nm, col)
            st[v] = (mu, sg); ymax[v] = max(ymax.get(v, 0), ym)
        stats[label] = st
    for a, v in ((ax[0][0], 'x'), (ax[0][1], 'y'), (ax[1][0], 'cotA'), (ax[1][1], 'cotB')):
        a.set_ylim(0.5, ymax[v] * 2)
        for k, (label, sd, col, _, _) in enumerate(loaded):
            mu, sg = stats[label][v]
            a.text(0.03, 0.90 - 0.085 * k, r"$\mu$=%.2f  $\sigma$=%.2f" % (mu, sg),
                   transform=a.transAxes, fontsize=7, color=col)
    fig.suptitle(suptitle.replace('overlay', 'pull overlay'), fontsize=11)
    fig.legend(handles=[Line2D([], [], color=c, lw=1.6, label=(f"{l}  \u00b7  s{s}" if legend_nll else l))
                        for l, s, c, _, _ in loaded],
               loc='lower center', ncol=ncol, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.05 if ncol == 2 else -0.015))
    fig.tight_layout(rect=[0, 0.09 if ncol == 2 else 0.035, 1, 0.95])
    pp = os.path.join(OUT, f'{name}_pull.png')
    fig.savefig(pp, dpi=140, facecolor='white'); plt.close(fig)

    # ---- per-bin bias in units of each arm's own sigma (the MDMM quantity) ----
    bias = {}
    for label, sd, col, df, nll in loaded:
        b = {}
        for v, tv, sc, ang in (('x','xtrue',SCALE[0],False), ('y','ytrue',SCALE[1],False),
                               ('cotA','cotAtrue',SCALE[2],True), ('cotB','cotBtrue',SCALE[3],True)):
            if ang:
                a = inverse_cot(df[tv].values*sc)*180/pi
                r = a - inverse_cot(df[v].values*sc)*180/pi
            else:
                a = df[tv].values*sc; r = a - df[v].values*sc
            sg = float(np.std(r)); lo, hi = a.min(), a.max(); step = (hi-lo)/15
            mb = [float(np.mean(r[(a > lo+i*step) & (a < lo+(i+1)*step)]))
                  for i in range(15) if ((a > lo+i*step) & (a < lo+(i+1)*step)).any()]
            b[v] = {'max_abs_bias_sigma': float(np.max(np.abs(mb))/sg),
                    'n_viol_0p17': int(np.sum(np.abs(np.array(mb))/sg > 0.17)), 'nbins': len(mb)}
        bias[label] = b
        print(f"  {label:26s} max|bias|/sig  " + "  ".join(
            f"{v}={b[v]['max_abs_bias_sigma']:.2f}({b[v]['n_viol_0p17']}/{b[v]['nbins']})"
            for v in ('x','y','cotA','cotB')), flush=True)

    import json
    json.dump({l: {'seed': s, 'nll': n, 'resid_std': res[l], 'pull': stats[l], 'bias': bias[l]}
               for l, s, c, _, n in loaded},
              open(os.path.join(OUT, f'{name}_stats.json'), 'w'), indent=1, default=float)
    print(" ", sp); print(" ", pp)


if __name__ == '__main__':
    want = sys.argv[1:] or ['overlay4', 'bimodal']
    for name in want:
        title, arms = SETS[name]
        print(f"\n=== {name} ===", flush=True)
        render(name, title, arms, legend_nll=(name != 'bimodal'))
