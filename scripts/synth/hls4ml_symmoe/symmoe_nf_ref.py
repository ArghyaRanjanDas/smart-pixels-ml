"""digi16-nf (SymMoE_digi16nf) reference models for the hardware port.

  float_ref(codes, W)        float64 numpy replica of the Keras model (codes in, 14 outputs), to pin the semantics
  fold_params(W)             trainable parameters -> the folded per-port values (still float)
  quantise_ports(Pf, cfg)    folded values -> integers on each port's ap_fixed grid  (the config image)
  fixed_ref(codes, Q, cfg)   THE bit-exact integer reference of the HLS kernel (nnet_symmoe_nf.h); int64 numpy
  tables(cfg)                the constant ROM tables (ln, 2^f, tanh) shared by the reference and the C++ kernel
  nll_v2(labels, pred, bsz)  custom_loss_v2 (softplus diag, batch sums) in float64 numpy

Fixed-point conventions (identical in nnet_symmoe_nf.h):
  * a value v at F fractional bits is the integer v*2^F.  '>>' is an arithmetic (floor) shift.
  * tdiv(n, d, s) = sign(n) * floor(|n| * 2^s / d) (truncation toward zero), 0 when d == 0.  This is how every
    division of the model is done: a correctly truncated quotient at the chosen precision (the C++ computes it with
    ONE reciprocal m = ceil(2^S/d) per denominator and a multiply, which is exact for |n| < 2^(S - bits(d)),
    Granlund-Montgomery; the widths are asserted in the C++).
  * every truncation to a narrower type is a floor shift followed by a saturation to the type's range.
"""
import math
import numpy as np

LN2 = math.log(2.0)
LABELS_SCALE = np.array([123.730140686, 30.942258835, 6.587896824, 1.848129153])
PX, PY, THICK = 50.0, 12.5, 100.0
N_EXP = 4

# ------------------------------------------------------------------ float replica


def _sp(x):
    return np.logaddexp(0.0, x)


def float_ref(codes, W, return_parts=False):
    c = codes.astype(np.float64)                      # (B,16,16,2)  axis1 = rows (y), axis2 = cols (x)
    B = len(c)
    q = c.sum(-1)
    px = q.sum(2)                                     # row-indexed  (RouterFeatures 'px')
    py = q.sum(1)                                     # col-indexed  (RouterFeatures 'py')
    tot = py.sum(1, keepdims=True) + 1e-9
    pxn, pyn = px / tot, py / tot
    yc = np.arange(16) - 7.5
    mu = (pyn * yc).sum(1, keepdims=True)
    d = yc[None] - mu
    m2 = (pyn * d ** 2).sum(1, keepdims=True); m3 = (pyn * d ** 3).sum(1, keepdims=True)
    std = np.sqrt(m2 + 1.0); skew = m3 / std ** 3
    py0 = c[..., 0].sum(1); py1 = c[..., -1].sum(1)
    c0 = (py0 * yc).sum(1, keepdims=True) / (py0.sum(1, keepdims=True) + 1e-9)
    c1 = (py1 * yc).sum(1, keepdims=True) / (py1.sum(1, keepdims=True) + 1e-9)
    tasym = c1 - c0
    logq = np.log(tot + 1.0)
    f = np.clip(np.concatenate([pxn, pyn, std, skew, tasym, logq], 1), -8, 8)
    h = np.maximum(f @ W['router_hidden/router_hidden/kernel'] + W['router_hidden/router_hidden/bias'], 0)
    z = h @ W['router_out/router_out/kernel'] + W['router_out/router_out/bias']
    e = np.exp(z - z.max(1, keepdims=True)); w = e / e.sum(1, keepdims=True)
    # ansatz (shared statistics)
    xcen = (np.arange(16) - 7.5) * PX; ycen = (np.arange(16) - 7.5) * PY
    prof_x = q.sum(1); prof_y = q.sum(2)
    sx = prof_x.sum(1); sy = prof_y.sum(1)
    xb = (prof_x * xcen).sum(1) / (sx + 1e-6); yb = (prof_y * ycen).sum(1) / (sy + 1e-6)
    m = (q >= 2.0 - 1e-6).astype(np.float64)
    cm = c * m[..., None]

    def cen(q2d, ax, cs):
        p = q2d.sum(ax); return (p * cs).sum(1) / (p.sum(1) + 1e-6)
    Tx = np.clip(cen(cm[..., 1], 1, xcen) - cen(cm[..., 0], 1, xcen), -800, 800)
    Ty = np.clip(cen(cm[..., 1], 2, ycen) - cen(cm[..., 0], 2, ycen), -200, 200)
    wx = (m.sum(1) > 0).sum(1).astype(np.float64); wy = (m.sum(2) > 0).sum(1).astype(np.float64)
    mus = []
    for k in range(N_EXP):
        g = lambda n: float(W[f'physics_ansatz_{k}/physics_ansatz_{k}/{n}'])
        aS = W[f'physics_ansatz_{k}/physics_ansatz_{k}/aff_scale']; aB = W[f'physics_ansatz_{k}/physics_ansatz_{k}/aff_bias']
        ls = W[f'physics_ansatz_{k}/physics_ansatz_{k}/labels_scale']
        th = g('theta_L')
        sa = np.tanh(np.clip(g('sign_k_alpha') * (g('sign_a0_alpha') - Tx), -30, 30))
        sb = np.tanh(np.clip(g('sign_k_beta') * (g('sign_a0_beta') - Ty), -30, 30))
        cota = np.maximum(wx - 1, 0) * PX / THICK * sa
        cotb = (np.maximum(wy - 1, 0) * PY / THICK + abs(g('lorentz_scale') * math.tan(th))) * sb
        yy = yb + THICK * math.tan(th) / 2.0
        raw = np.stack([xb, yy, cota, cotb], 1) / ls
        mus.append(aS * raw + aB)
    mean = sum(w[:, k:k + 1] * mus[k] for k in range(N_EXP))
    # error MLP
    tm = c.mean(-1)
    prx = tm.mean(2); pry = tm.mean(1)                # AvgPool (1,16) -> row-indexed; (16,1) -> col-indexed
    hh = np.tanh(np.concatenate([prx, pry], 1) @ W['error_mlp/hidden/kernel'] + W['error_mlp/hidden/bias'])
    chol = hh @ W['error_mlp/chol_entries/kernel'] + W['error_mlp/chol_entries/bias']
    out = np.concatenate([np.stack([mean[:, 0], chol[:, 0], mean[:, 1], chol[:, 1], mean[:, 2], chol[:, 2],
                                    mean[:, 3], chol[:, 3]], 1), chol[:, 4:]], 1)
    if return_parts:
        return out, dict(f=f, h=h, z=z, w=w, Tx=Tx, Ty=Ty, xb=xb, yb=yb, wx=wx, wy=wy, mus=np.stack(mus, 1),
                         eh=np.concatenate([prx, pry], 1) @ W['error_mlp/hidden/kernel'] + W['error_mlp/hidden/bias'],
                         chol=chol, mean=mean)
    return out


# ------------------------------------------------------------------ NLL (custom_loss_v2, numpy)


def nll_v2(y, p, bsz=None):
    """custom_loss_v2 per event (float64); returns (mean of batch sums, per-event array)."""
    y = y.astype(np.float64); p = p.astype(np.float64)
    mu = p[:, [0, 2, 4, 6]]
    diag = 1e-9 + _sp(p[:, [1, 3, 5, 7]])
    cov = p[:, 8:14]
    L = np.zeros((len(p), 4, 4))
    for i in range(4):
        L[:, i, i] = diag[:, i]
    L[:, 1, 0] = cov[:, 0]; L[:, 2, 0] = cov[:, 1]; L[:, 2, 1] = cov[:, 2]
    L[:, 3, 0] = cov[:, 3]; L[:, 3, 1] = cov[:, 4]; L[:, 3, 2] = cov[:, 5]
    r = y - mu
    z = np.zeros_like(r)
    for i in range(4):
        z[:, i] = (r[:, i] - (L[:, i, :i] * z[:, :i]).sum(1)) / L[:, i, i]
    per = (0.5 * z ** 2 + np.log(diag) + 0.5 * math.log(2 * math.pi)).sum(1)
    if bsz is None:
        return float(per.sum()), per
    s, o = [], 0
    for b in bsz:
        s.append(per[o:o + b].sum()); o += b
    return float(np.mean(s)), per


# ------------------------------------------------------------------ fixed-point configuration

DEFAULT_CFG = dict(
    FR=14,            # router feature frac bits (features ap_fixed<4+FR,4>, range [-8,8))
    FM=18,            # raw-moment frac bits (mu, E[y^2], E[y^3], m2, m3, std)
    FH=10, IH=6,      # router hidden (after relu) ap_ufixed<IH+FH, IH>
    FZ=10, IZ=7,      # router logits ap_fixed<IZ+FZ, IZ>
    FU=10,            # softmax: u = (z - zmax) * log2(e) frac bits
    TE=5,             # softmax: 2^f table index bits (+ linear interpolation on the remaining FU-TE bits)
    FE=16,            # softmax: e_k = 2^u frac bits
    FW=16,            # router weights w_k ap_ufixed<1+FW,1>
    FT=12,            # drift centroid quotients G/D frac bits (half-pixel units, range +-15)
    FA=10, IA=6,      # sign-gate tanh argument ap_fixed<IA+FA, IA> (saturating)
    TT=5, TR=8,       # tanh table: step 2^-TT on [0, TR), linear interpolation, saturates to the last entry
    FS=16,            # tanh / sign outputs frac bits
    FMU=16,           # expert means frac bits
    FO=14, IO=8,      # output ap_fixed<IO+FO, IO>
    FEH=10, IEH=6,    # error-MLP hidden pre-activation ap_fixed<IEH+FEH, IEH> (saturating)
    LT=6,             # ln table index bits (mantissa), linear interpolation on the remaining 10-LT bits
    FL=18,            # ln table frac bits
    WB=16,            # default port width (bits) for weights / folded constants
)

# port list: name -> (shape, description).  Integer bits per port are chosen from the trained values with headroom
# (PORT_IBITS below); frac = width - ibits.
PORTS = [
    ('w_r1', (36, 16), 'router_hidden kernel'),
    ('b_r1', (16,), 'router_hidden bias'),
    ('w_r2', (16, 4), 'router_out kernel'),
    ('b_r2', (4,), 'router_out bias'),
    ('a_sx', (4,), 'expert x slope  = aff_scale0 * 50 / ls0           (multiplies mu_col = S1/(2 tot))'),
    ('a_bx', (4,), 'expert x bias   = aff_bias0'),
    ('a_sy', (4,), 'expert y slope  = aff_scale1 * 12.5 / ls1         (multiplies mu_row = R1/(2 tot))'),
    ('a_by', (4,), 'expert y bias   = aff_scale1 * T tan(theta_L)/2 / ls1 + aff_bias1  (Lorentz y offset)'),
    ('a_ka', (4,), 'sign gate a slope  = 25 * sign_k_alpha            (multiplies the x-drift in half-pixels)'),
    ('a_ca', (4,), 'sign gate a offset = sign_k_alpha * sign_a0_alpha'),
    ('a_kb', (4,), 'sign gate b slope  = 6.25 * sign_k_beta'),
    ('a_cb', (4,), 'sign gate b offset = sign_k_beta * sign_a0_beta'),
    ('a_sa', (4,), 'cot a slope = aff_scale2 * (50/100) / ls2           (per support column beyond the first)'),
    ('a_ba', (4,), 'cot a bias  = aff_bias2'),
    ('a_sb', (4,), 'cot b slope = aff_scale3 * (12.5/100) / ls3         (per support row beyond the first)'),
    ('a_lb', (4,), 'cot b Lorentz term = aff_scale3 * |lorentz_scale * tan(theta_L)| / ls3'),
    ('a_bb', (4,), 'cot b bias  = aff_bias3'),
    ('w_e1', (32, 8), 'error_mlp hidden kernel'),
    ('b_e1', (8,), 'error_mlp hidden bias'),
    ('w_e2', (8, 10), 'error_mlp chol_entries kernel'),
    ('b_e2', (10,), 'error_mlp chol_entries bias'),
]


def fold_params(W):
    P = {}
    P['w_r1'] = W['router_hidden/router_hidden/kernel']; P['b_r1'] = W['router_hidden/router_hidden/bias']
    P['w_r2'] = W['router_out/router_out/kernel']; P['b_r2'] = W['router_out/router_out/bias']
    P['w_e1'] = W['error_mlp/hidden/kernel']; P['b_e1'] = W['error_mlp/hidden/bias']
    P['w_e2'] = W['error_mlp/chol_entries/kernel']; P['b_e2'] = W['error_mlp/chol_entries/bias']
    cols = {k: [] for k in ('a_sx', 'a_bx', 'a_sy', 'a_by', 'a_ka', 'a_ca', 'a_kb', 'a_cb', 'a_sa', 'a_ba', 'a_sb',
                            'a_lb', 'a_bb')}
    for k in range(N_EXP):
        g = lambda n: np.asarray(W[f'physics_ansatz_{k}/physics_ansatz_{k}/{n}'], np.float64)
        aS, aB, ls = g('aff_scale'), g('aff_bias'), g('labels_scale')
        th = float(g('theta_L'))
        cols['a_sx'].append(aS[0] * PX / ls[0]); cols['a_bx'].append(aB[0])
        cols['a_sy'].append(aS[1] * PY / ls[1]); cols['a_by'].append(aS[1] * THICK * math.tan(th) / 2 / ls[1] + aB[1])
        cols['a_ka'].append(25.0 * g('sign_k_alpha')); cols['a_ca'].append(g('sign_k_alpha') * g('sign_a0_alpha'))
        cols['a_kb'].append(6.25 * g('sign_k_beta')); cols['a_cb'].append(g('sign_k_beta') * g('sign_a0_beta'))
        cols['a_sa'].append(aS[2] * (PX / THICK) / ls[2]); cols['a_ba'].append(aB[2])
        cols['a_sb'].append(aS[3] * (PY / THICK) / ls[3])
        cols['a_lb'].append(aS[3] * abs(float(g('lorentz_scale')) * math.tan(th)) / ls[3]); cols['a_bb'].append(aB[3])
    for k, v in cols.items():
        P[k] = np.array(v, np.float64)
    return {k: np.asarray(v, np.float64) for k, v in P.items()}


def port_formats(Pf, cfg, headroom=2.0, ibits_override=None):
    """(width, ibits) per port: ibits from max|value| * headroom (sign included), width = cfg['WB_<port>'] or cfg WB."""
    fm = {}
    for name, shape, _ in PORTS:
        WB = cfg.get('WB_' + name, cfg['WB'])
        v = np.abs(Pf[name]).max()
        need = math.floor(math.log2(max(v * headroom, 2 ** -8))) + 2      # sign + integer bits
        I = max(need, 1)
        if ibits_override and name in ibits_override:
            I = ibits_override[name]
        fm[name] = (WB, I)
    return fm


def quantise_ports(Pf, fm):
    """round-to-nearest onto each port's grid (this is the config image). Returns ints and the max abs rounding error."""
    Q, err = {}, {}
    for name, shape, _ in PORTS:
        Wd, I = fm[name]; F = Wd - I
        q = np.rint(np.asarray(Pf[name]) * 2.0 ** F).astype(np.int64)
        lo, hi = -(1 << (Wd - 1)), (1 << (Wd - 1)) - 1
        assert q.min() >= lo and q.max() <= hi, (name, q.min(), q.max(), Wd, I)
        Q[name] = q.reshape(shape); err[name] = float(np.abs(q * 2.0 ** -F - Pf[name]).max())
    return Q, err


# ------------------------------------------------------------------ tables


def tables(cfg):
    T = {}
    LT, FL = cfg['LT'], cfg['FL']
    T['ln'] = np.array([round(math.log1p(i / 2 ** LT) * 2 ** FL) for i in range(2 ** LT + 1)], np.int64)
    TE, FE = cfg['TE'], cfg['FE']
    T['exp2'] = np.array([round(2.0 ** (i / 2 ** TE) * 2 ** FE) for i in range(2 ** TE + 1)], np.int64)
    TT, TR, FS = cfg['TT'], cfg['TR'], cfg['FS']
    T['tanh'] = np.array([round(math.tanh(i / 2 ** TT) * 2 ** FS) for i in range(TR * 2 ** TT + 1)], np.int64)
    return T


# ------------------------------------------------------------------ fixed-point reference

def tdiv(n, d, s):
    """sign(n) * floor(|n| * 2^s / d); 0 where d == 0.  n, d int64 arrays (d >= 0)."""
    n = np.asarray(n, np.int64); d = np.asarray(d, np.int64)
    a = np.abs(n) << s
    assert (a >= 0).all() and a.max(initial=0) < 2 ** 62
    dd = np.where(d == 0, 1, d)
    q = a // dd
    return np.where(d == 0, 0, np.sign(n) * q)


def sat(x, W, I, signed=True):
    """x already at the target frac bits; saturate to ap_(u)fixed<W,I>."""
    if signed:
        lo, hi = -(1 << (W - 1)), (1 << (W - 1)) - 1
    else:
        lo, hi = 0, (1 << W) - 1
    return np.clip(x, lo, hi)


def shr(x, s):
    """floor shift by s (s may be negative = left shift)."""
    return x >> s if s >= 0 else x << (-s)


def isqrt(v):
    return np.array([math.isqrt(int(t)) for t in np.ravel(v)], np.int64).reshape(np.shape(v))


def interp(table, idx, r, rbits):
    """table[idx] + floor((table[idx+1]-table[idx]) * r / 2^rbits)"""
    a = table[idx]; b = table[idx + 1]
    return a + (((b - a) * r) >> rbits)


def tanh_fx(x, F_in, T, cfg):
    """tanh of x (int at F_in frac, |x| < 2^something) -> int at FS frac. odd; table on [0,TR) step 2^-TT, lin. interp."""
    TT, TR = cfg['TT'], cfg['TR']
    ax = np.abs(x)
    rb = F_in - TT                                  # remainder bits
    assert rb >= 0
    idx = ax >> rb
    r = ax & ((1 << rb) - 1)
    nmax = TR << TT
    sat_ = idx >= nmax
    idx_c = np.minimum(idx, nmax - 1)
    v = interp(T['tanh'], idx_c, r, rb)
    v = np.where(sat_, T['tanh'][nmax], v)
    return np.where(x < 0, -v, v)


def fixed_ref(codes, Q, fm, cfg, return_parts=False):
    c = codes.astype(np.int64)
    B = len(c)
    FR, FM = cfg['FR'], cfg['FM']
    T = tables(cfg)
    ar = np.arange(16, dtype=np.int64); o2 = 2 * ar - 15            # 2*(i - 7.5): odd integers -15..15
    q = c.sum(-1)                                                   # (B,16,16) 0..6
    Prow = q.sum(2); Pcol = q.sum(1)                                # row-indexed / col-indexed profiles 0..96
    tot = Pcol.sum(1)                                               # 0..1536
    S1 = (Pcol * o2).sum(1); S2 = (Pcol * o2 ** 2).sum(1); S3 = (Pcol * o2 ** 3).sum(1)
    R1 = (Prow * o2).sum(1)
    p0 = c[..., 0].sum(1); p1 = c[..., 1].sum(1)                    # col-indexed per slice
    T0 = p0.sum(1); T1 = p1.sum(1)
    Q0 = (p0 * o2).sum(1); Q1 = (p1 * o2).sum(1)
    msk = (q >= 2).astype(np.int64)
    cm0 = c[..., 0] * msk; cm1 = c[..., 1] * msk
    D0 = cm0.sum((1, 2)); D1 = cm1.sum((1, 2))
    G0 = (cm0.sum(1) * o2).sum(1); G1 = (cm1.sum(1) * o2).sum(1)    # col (x) drift numerators
    H0 = (cm0.sum(2) * o2).sum(1); H1 = (cm1.sum(2) * o2).sum(1)    # row (y) drift numerators
    wx = (msk.sum(1) > 0).sum(1); wy = (msk.sum(2) > 0).sum(1)

    # ---- router features (FR frac), all clipped to [-8, 8 - 2^-FR]
    fhi, flo = (8 << FR) - 1, -(8 << FR)
    pxn = tdiv(Prow, tot[:, None], FR); pyn = tdiv(Pcol, tot[:, None], FR)
    A1 = tdiv(S1, tot, FM - 1)              # mu   = S1 / (2 tot)
    A2 = tdiv(S2, tot, FM - 2)              # E y^2 = S2 / (4 tot)
    A3 = tdiv(S3, tot, FM - 3)              # E y^3 = S3 / (8 tot)
    sq = (A1 * A1) >> FM
    m2 = A2 - sq
    m3 = A3 - ((3 * A1 * A2) >> FM) + ((2 * sq * A1) >> FM)
    v = np.maximum(m2, 0) + (1 << FM)
    std = isqrt(v << FM)                    # FM frac
    std3 = (((std * std) >> FM) * std) >> FM
    skew = tdiv(m3, std3, FR)               # FR frac
    c0 = tdiv(Q0, T0, FR - 1); c1 = tdiv(Q1, T1, FR - 1)
    tasym = c1 - c0
    # ln(tot + 1): n = tot+1 in 1..1537 (11 bits); e = floor(log2 n); mantissa f = 10 bits, exact
    n = tot + 1
    e = np.floor(np.log2(n)).astype(np.int64)
    e = np.where((1 << (e + 1)) <= n, e + 1, e); e = np.where((1 << e) > n, e - 1, e)
    f = ((n << (10 - e)) & 1023)
    LT, FL = cfg['LT'], cfg['FL']
    lnm = interp(T['ln'], f >> (10 - LT), f & ((1 << (10 - LT)) - 1), 10 - LT)
    LN2_FL = round(LN2 * 2 ** FL)
    logq = shr(e * LN2_FL + lnm, FL - FR)
    feats = np.concatenate([pxn, pyn, shr(std, FM - FR)[:, None], skew[:, None], tasym[:, None], logq[:, None]], 1)
    feats = np.clip(feats, flo, fhi)

    # ---- router MLP
    def Fq(name):
        return fm[name][0] - fm[name][1]
    acc = feats @ Q['w_r1'] + shr(Q['b_r1'], Fq('b_r1') - (FR + Fq('w_r1')))      # frac FR + Fw
    FH, IH = cfg['FH'], cfg['IH']
    h = sat(shr(np.maximum(acc, 0), FR + Fq('w_r1') - FH), IH + FH, IH, signed=False)
    FZ, IZ = cfg['FZ'], cfg['IZ']
    acc2 = h @ Q['w_r2'] + shr(Q['b_r2'], Fq('b_r2') - (FH + Fq('w_r2')))
    z = sat(shr(acc2, FH + Fq('w_r2') - FZ), IZ + FZ, IZ)
    # ---- softmax: e_k = 2^((z_k - zmax) log2 e), w_k = e_k / sum e
    FU, TE, FE, FW = cfg['FU'], cfg['TE'], cfg['FE'], cfg['FW']
    L2E = round(math.log2(math.e) * 2 ** 16)
    dz = z - z.max(1, keepdims=True)                                 # <= 0, FZ frac
    u = (dz * L2E) >> (FZ + 16 - FU)                                 # FU frac, <= 0 (floor)
    ni = u >> FU                                                     # integer part (<= 0)
    fr = u & ((1 << FU) - 1)                                         # [0, 2^FU)
    m_ = interp(T['exp2'], fr >> (FU - TE), fr & ((1 << (FU - TE)) - 1), FU - TE)   # 2^frac at FE
    sh = np.minimum(-ni, 63)
    ek = np.where(sh > FE + 1, 0, m_ >> np.minimum(sh, FE + 1))      # 2^u at FE frac
    se = ek.sum(1, keepdims=True)                                    # >= 2^FE (the max term is exactly 1.0)
    w = tdiv(ek, se, FW)                                             # FW frac, in [0, 1]

    # ---- ansatz statistics (shared by the 4 experts)
    FT, FA, IA, FS, FMU = cfg['FT'], cfg['FA'], cfg['IA'], cfg['FS'], cfg['FMU']
    B1 = tdiv(R1, tot, FM - 1)                                       # row centroid (pixels from centre)
    ux0 = tdiv(G0, D0, FT); ux1 = tdiv(G1, D1, FT)                   # masked col centroid, half-pixel units
    uy0 = tdiv(H0, D0, FT); uy1 = tdiv(H1, D1, FT)
    dux = ux1 - ux0; duy = uy1 - uy0                                 # FT frac, |.| <= 30
    wx1 = np.maximum(wx - 1, 0); wy1 = np.maximum(wy - 1, 0)
    amax, amin = (1 << (IA + FA - 1)) - 1, -(1 << (IA + FA - 1))
    mus = np.zeros((B, N_EXP, 4), np.int64)
    sga = np.zeros((B, N_EXP), np.int64); sgb = np.zeros((B, N_EXP), np.int64)
    for k in range(N_EXP):
        # x, y (FMU frac)
        mx = shr(Q['a_sx'][k] * A1, Fq('a_sx') + FM - FMU) + shr(Q['a_bx'][k], Fq('a_bx') - FMU)
        my = shr(Q['a_sy'][k] * B1, Fq('a_sy') + FM - FMU) + shr(Q['a_by'][k], Fq('a_by') - FMU)
        # sign gates: arg = c - k * drift  (FA frac, saturating), tanh -> FS frac
        arga = shr(Q['a_ca'][k], Fq('a_ca') - FA) - shr(Q['a_ka'][k] * dux, Fq('a_ka') + FT - FA)
        argb = shr(Q['a_cb'][k], Fq('a_cb') - FA) - shr(Q['a_kb'][k] * duy, Fq('a_kb') + FT - FA)
        arga = np.clip(arga, amin, amax); argb = np.clip(argb, amin, amax)
        ta = tanh_fx(arga, FA, T, cfg); tb = tanh_fx(argb, FA, T, cfg)
        sga[:, k] = ta; sgb[:, k] = tb
        maga = wx1 * Q['a_sa'][k]                                    # Fq(a_sa) frac
        magb = wy1 * Q['a_sb'][k] + shr(Q['a_lb'][k], Fq('a_lb') - Fq('a_sb'))
        ca = shr(maga * ta, Fq('a_sa') + FS - FMU) + shr(Q['a_ba'][k], Fq('a_ba') - FMU)
        cb = shr(magb * tb, Fq('a_sb') + FS - FMU) + shr(Q['a_bb'][k], Fq('a_bb') - FMU)
        mus[:, k] = np.stack([mx, my, ca, cb], 1)
    FO, IO = cfg['FO'], cfg['IO']
    mean = sat(shr((w[:, :, None] * mus).sum(1), FW + FMU - FO), IO + FO, IO)   # (B,4) FO frac

    # ---- error MLP: inputs = profiles / 32 (5 frac bits, exact)
    xin = np.concatenate([Prow, Pcol], 1)                            # 5 frac
    FEH, IEH = cfg['FEH'], cfg['IEH']
    acc3 = xin @ Q['w_e1'] + shr(Q['b_e1'], Fq('b_e1') - (5 + Fq('w_e1')))
    eh = sat(shr(acc3, 5 + Fq('w_e1') - FEH), IEH + FEH, IEH)
    th = tanh_fx(eh, FEH, T, cfg)                                    # FS frac
    acc4 = th @ Q['w_e2'] + shr(Q['b_e2'], Fq('b_e2') - (FS + Fq('w_e2')))
    chol = sat(shr(acc4, FS + Fq('w_e2') - FO), IO + FO, IO)
    out = np.concatenate([np.stack([mean[:, 0], chol[:, 0], mean[:, 1], chol[:, 1], mean[:, 2], chol[:, 2],
                                    mean[:, 3], chol[:, 3]], 1), chol[:, 4:]], 1)
    if return_parts:
        return out, dict(feats=feats, h=h, z=z, w=w, mus=mus, sga=sga, sgb=sgb, eh=eh, th=th, mean=mean, chol=chol,
                         acc=acc, acc2=acc2, acc3=acc3, dux=dux, duy=duy, A1=A1, B1=B1, std=std, m3=m3, std3=std3)
    return out
