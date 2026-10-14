"""Per-arm training curves, summary and pull plots for the O25 family.

Cannot reuse make_perf_plots_o22.py: that one is bound to the ViT pair-lattice
router (reads simple_router_output.selected_indices) and feeds the full 101-slice
generator. The O25 arms are QConv2D/Conv2D on the PRE-SLICED 2-slice cache with a
frozen readout and no router at all. The residual/pull maths and the corrected
Gaussian bin-centre fit are ported verbatim.
"""
import os, sys, json, gc
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
R = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(R, 'two_bit_optimization_helpers'))
import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.set_logical_device_configuration(
        g, [tf.config.LogicalDeviceConfiguration(memory_limit=4000)])
    except Exception: pass
from conditional_nll import custom_loss_v2, cot_to_deg
from models.conv2d_model_quantized import QConv2D_Max_SoftQuantizer
from models.conv2d_model_nonquantized import Conv2D_Max_SoftQuantizer

OUT = os.environ.get('SMARTPIX_OUT', os.path.join(R, 'runs', 'perf_plots_o25'))
os.makedirs(OUT, exist_ok=True)
THR = [8.565593719482422, 20.227340698242188, 47.49168395996094]
LEVELS = np.array([0., 1., 2., 3.], np.float32)
BASE = ("/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_"
        "10ps_300k_convolved_to_200ps/shuffled_3d")
meta = json.load(open(os.path.join(BASE, "TFR_files_all101_noise_contained_discovery",
                                   "TFR_test", "metadata.json")))
SCALE = np.asarray(meta["labels_scale"], dtype=np.float64)

# (label, run root, model kind, n_filters)
ARMS = [
 ("quant+constrained (O25)", "runs/o25_qconv2d",             "q", 5),
 ("quant+unconstrained",     "runs/o25_plain",               "q", 5),
 ("float+constrained",       "runs/o25_float",               "f", 5),
 ("float+unconstrained",     "runs/o25_float_plain",         "f", 5),
 ("distill narrow KL",       "runs/o25_distill/narrow_kl",   "q", 5),
 ("distill wide KL",         "runs/o25_distill/wide_kl",     "q", 16),
 ("wide control (no KL)",    "runs/o25_distill/wide_ctrl",   "q", 16),
 ("real-tanh (rejected)",    "runs/o25_realtanh",            "qr", 5),
]
z = np.load(os.path.join(R, 'runs/o25_qconv2d/_preslice_cache/preslice_val_10_21.npz'))
NB = int(z['n'])
XB = [z[f'X{i}'] for i in range(NB)]; YB = [z[f'y{i}'] for i in range(NB)]

def build(kind, nf):
    # 'qr' = quantized, trained under set_internal_sigmoid('real'). Rebuilding it
    # under the default 'hard' is a DIALECT MISMATCH and scores ~1e13 -- the same
    # class of artefact as scoring a softplus checkpoint under the relu mapping.
    from qkeras.quantizers import set_internal_sigmoid
    set_internal_sigmoid('real' if kind == 'qr' else 'hard')
    fn = Conv2D_Max_SoftQuantizer if kind == 'f' else QConv2D_Max_SoftQuantizer
    return fn((16,16,2), n_filters=nf, pool_size=3, initial_thresholds=THR,
              threshold_offset=0.0, initial_levels=LEVELS)

def predict_df(ck, kind, nf):
    tf.keras.backend.clear_session(); gc.collect()
    m = build(kind, nf); m.load_weights(ck)
    P = np.concatenate([np.asarray(m.predict_on_batch(x)) for x in XB], 0)
    Y = np.concatenate([np.asarray(y) for y in YB], 0)
    nll = float(custom_loss_v2(tf.constant(Y, tf.float32),
                               tf.constant(P, tf.float32))) / max(1, NB)
    cols = ['x','M11','y','M22','cotA','M33','cotB','M44','M21','M31','M32','M41','M42','M43']
    df = pd.DataFrame(P, columns=cols)
    df['xtrue'],df['ytrue'],df['cotAtrue'],df['cotBtrue'] = Y[:,0],Y[:,1],Y[:,2],Y[:,3]
    # softplus diagonal, matching custom_loss_v2
    for mm in ('M11','M22','M33','M44'):
        v = df[mm].values
        df[mm] = 1e-9 + np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0.0)
    df['sigmax']    = np.abs(df['M11'])
    df['sigmay']    = np.sqrt(df['M21']**2 + df['M22']**2)
    df['sigmacotA'] = np.sqrt(df['M31']**2 + df['M32']**2 + df['M33']**2)
    df['sigmacotB'] = np.sqrt(df['M41']**2 + df['M42']**2 + df['M43']**2 + df['M44']**2)
    for v,t,s in (('x','xtrue','sigmax'),('y','ytrue','sigmay'),
                  ('cotA','cotAtrue','sigmacotA'),('cotB','cotBtrue','sigmacotB')):
        df['pull'+v] = (df[t]-df[v])/df[s]
        df['res'+v]  = (df[v]-df[t])
    del m; gc.collect()
    return df, nll

def gauss(x, A, mu, sigma): return A*np.exp(-(x-mu)**2/(2*sigma**2))

def pull_ax(ax, df, var, name):
    h = ax.hist(df['pull'+var], bins=np.linspace(-5,5,50), histtype='step', color='#2b6cb0')
    ax.set_xlabel(name); ax.set_yscale('log')
    # bin CENTRES: linspace(-5,5,50) -> 49 bins of width 10/49, offset 0.1020.
    _e = h[1]; xd = _e[:-1] + (_e[1]-_e[0])/2.0; yd = h[0]
    try:
        p,_ = curve_fit(gauss, xd, yd, p0=[max(yd.max(),1),0,1], maxfev=10000)
        mu, sg = p[1], abs(p[2])
        xb = np.linspace(-5,5,200); ax.plot(xb, gauss(xb,*p), color='black', lw=1)
    except Exception:
        mu, sg = float(df['pull'+var].mean()), float(df['pull'+var].std())
    ax.set_ylim(0.5, max(yd.max()*2,10))
    ax.text(0.04,0.90,r"$\mu$=%.2f"%mu,transform=ax.transAxes,fontsize=9)
    ax.text(0.04,0.80,r"$\sigma$=%.2f"%sg,transform=ax.transAxes,fontsize=9)
    return mu, sg


# ---------------------------------------------------------------------------
# Ported VERBATIM from make_perf_plots_o22.py so the O25 slides carry the same
# figure language as the rest of the deck: a seaborn binned residual profile
# with a fill_between band of the model's OWN predicted sigma per bin, and for
# the angles the asymmetric sigma propagated through inverse_cot into degrees.
# My earlier plain histograms were a different plot, not a restyled one.
# ---------------------------------------------------------------------------
import seaborn as sns
pi = np.pi

def inverse_cot(c):
    a = np.arctan(1.0/c); return np.where(a < 0, a + pi, a)

def resid_pos(ax, df, v1, v2, name, scaling):
    a = df[v1].values*scaling; b = df[v2].values*scaling; res = a - b
    nb = 15; xmin, xmax = np.min(a), np.max(a); step = (xmax-xmin)/nb
    sns.regplot(x=a, y=res, x_bins=np.linspace(xmin, xmax, nb), fit_reg=None, marker='.', ax=ax)
    ax.set_xlabel('True '+name); ax.set_ylabel('True - pred '+name)
    ax.axhline(0, ls='--', lw=1.0, c='#334155')
    sig = df['sigma'+v2].values*scaling
    up, dn = [], []
    for i in range(nb):
        m = (a > xmin+i*step) & (a < xmin+(i+1)*step)
        mu = np.mean(res[m]) if m.any() else 0.0; sg = np.mean(sig[m]) if m.any() else 0.0
        up.append(mu+sg); dn.append(mu-sg)
    ax.fill_between(np.linspace(xmin, xmax, nb), up, dn, alpha=0.2)
    return float(np.std(res))

def resid_ang(ax, df, v1, v2, name, scaling):
    ang  = inverse_cot(df[v2].values*scaling)*180/pi
    angT = inverse_cot(df[v1].values*scaling)*180/pi
    aup = np.abs(inverse_cot((df[v2].values + df['sigma'+v2].values)*scaling)*180/pi - ang)
    adn = np.abs(inverse_cot((df[v2].values - df['sigma'+v2].values)*scaling)*180/pi - ang)
    nb = 15; xmin, xmax = np.min(angT), np.max(angT); step = (xmax-xmin)/nb
    res = angT - ang
    sns.regplot(x=angT, y=res, x_bins=np.linspace(xmin, xmax, nb), fit_reg=None, marker='.', ax=ax)
    ax.set_xlabel('True '+name); ax.set_ylabel('True - pred '+name)
    ax.axhline(0, ls='--', lw=1.0, c='#334155')
    up, dn = [], []
    for i in range(nb):
        m = (angT > xmin+i*step) & (angT < xmin+(i+1)*step)
        mu = np.mean(res[m]) if m.any() else 0.0
        up.append(mu+(np.mean(aup[m]) if m.any() else 0.0))
        dn.append(mu-(np.mean(adn[m]) if m.any() else 0.0))
    ax.fill_between(np.linspace(xmin, xmax, nb), up, dn, alpha=0.2)
    return float(np.std(res))

def make_summary(df, title, nll, path):
    fig, ax = plt.subplots(2, 2, figsize=(9, 7)); fig.tight_layout(pad=4.0)
    fig.suptitle(f"Summary - {title}   (val plain NLL {nll:,.0f})", fontsize=11)
    r = {}
    r['x']    = resid_pos(ax[0][0], df, 'xtrue', 'x', r'$x$ [$\mu$m]', SCALE[0])
    r['y']    = resid_pos(ax[0][1], df, 'ytrue', 'y', r'$y$ [$\mu$m]', SCALE[1])
    r['cotA'] = resid_ang(ax[1][0], df, 'cotAtrue', 'cotA', r'$\alpha$ [deg]', SCALE[2])
    r['cotB'] = resid_ang(ax[1][1], df, 'cotBtrue', 'cotB', r'$\beta$ [deg]', SCALE[3])
    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return r

def make_pull(df, title, nll, path):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 6))
    fig.suptitle(f"Pull - {title}   (val plain NLL {nll:,.0f})", fontsize=11)
    out=[]
    for a,v,nm in ((ax[0][0],'x',r'$x$ pull'), (ax[0][1],'y',r'$y$ pull'),
                   (ax[1][0],'cotA',r'$\cot\alpha$ pull'), (ax[1][1],'cotB',r'$\cot\beta$ pull')):
        mu,sg = pull_ax(a, df, v, nm); out.append((v,mu,sg))
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(path, dpi=140, facecolor='white'); plt.close(fig)
    return out

UNITS = {'x':'x [um]','y':'y [um]','cotA':r'cot$\alpha$','cotB':r'cot$\beta$'}
VARS  = ['x','y','cotA','cotB']

def seeds_of(root):
    d = os.path.join(R, root)
    if not os.path.isdir(d): return []
    out=[]
    for s in sorted(os.listdir(d)):
        if not s.startswith('s'): continue
        h = os.path.join(d,s,f'seed_{s[1:]}','history.csv')
        w = os.path.join(d,s,f'seed_{s[1:]}','best.weights.hdf5')
        if os.path.exists(h) and os.path.exists(w): out.append((s[1:],h,w))
    return out

def best_of(h):
    df = pd.read_csv(h)
    c = 'val_plain_nll' if 'val_plain_nll' in df else 'val_loss'
    return df, df[c].min(), c

def make_curve(label, root, path):
    ss = seeds_of(root)
    if not ss: return None
    # 2.31 aspect to fill the deck's 1611x697 single-image box exactly
    fig, ax = plt.subplots(figsize=(11,4.76))
    for sd,h,_ in ss:
        df,b,c = best_of(h)
        ax.plot(df['epoch'], df[c].rolling(25,min_periods=1).min(), lw=1.2, label=f"s{sd}  {b:,.0f}")
    ax.set_xlabel('epoch'); ax.set_ylabel('val plain NLL (best-so-far)')
    ax.set_title(f"{label} — training curves"); ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path

if __name__ == '__main__':
    import numpy as _np
    rows=[]
    for label, root, kind, nf in ARMS:
        ss = seeds_of(root)
        if not ss:
            print(f"SKIP {label}: no finished seeds under {root}", flush=True); continue
        tag = root.replace('runs/','').replace('/','_')
        make_curve(label, root, os.path.join(OUT, f'curve_{tag}.png'))
        for sd,h,w in ss:                       # EVERY seed, not just the best
            df, nll = predict_df(w, kind, nf)
            ppath=os.path.join(OUT,f'pull_{tag}_s{sd}.png')
            spath=os.path.join(OUT,f'summary_{tag}_s{sd}.png')
            st=make_pull(df, f"{label} · seed {sd}", nll, ppath)
            make_summary(df, f"{label} · seed {sd}", nll, spath)
            # resolutions in the units the deck's tables use: x,y in um (label
            # scale) and the angles in DEGREES via cot_to_deg, matching the MDMM
            # constraint space -- cot-space and degree-space widths differ.
            sx=float((df['resx'].values*SCALE[0]).std())
            sy=float((df['resy'].values*SCALE[1]).std())
            da=cot_to_deg(tf.constant(df['cotA'].values,tf.float32),float(SCALE[2])).numpy() - \
               cot_to_deg(tf.constant(df['cotAtrue'].values,tf.float32),float(SCALE[2])).numpy()
            db=cot_to_deg(tf.constant(df['cotB'].values,tf.float32),float(SCALE[3])).numpy() - \
               cot_to_deg(tf.constant(df['cotBtrue'].values,tf.float32),float(SCALE[3])).numpy()
            best=float(pd.read_csv(h)['val_plain_nll'].min())
            rows.append(dict(arm=label, tag=tag, seed=sd, nll=nll, best_hist=best,
                             sx=sx, sy=sy, sa=float(_np.std(da)), sb=float(_np.std(db)),
                             **{f'pull_{v}_mu':mu for v,mu,_ in st},
                             **{f'pull_{v}_sd':sg for v,_,sg in st}))
            print(f"OK {label:28s} s{sd}  nll {nll:10,.0f}  sx {sx:6.2f} sy {sy:6.2f} "
                  f"sa {_np.std(da):6.2f} sb {_np.std(db):6.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'o25_seed_table.csv'), index=False)
    print("\nwrote", os.path.join(OUT,'o25_seed_table.csv'))
