// digi16-nf (SymMoE_digi16nf) HLS kernel for hls4ml (Vitis backend), one call = one event.
//
// THE bit-exact twin of symmoe_nf_ref.fixed_ref() (scripts/synth/hls4ml_symmoe/symmoe_nf_ref.py): every
// operation, every floor shift, every saturation and every table interpolation is the same integer operation.
// Values are carried as raw integers (a value v at F fractional bits is the integer v*2^F); the trained
// parameters arrive on ap_fixed ports and are read as their raw bits.  All intermediate widths are WORST-CASE
// for any port contents (the design is programmable), computed with constexpr from the CONFIG_T constants.
//
// Divisions: every division of the model is a correctly truncated quotient sign(n)*floor(|n|*2^s/d) computed
// by an unrolled restoring divider whose quotient width QB is an analytic bound (n < d*2^QB holds for every
// possible input, e.g. Prow <= tot), so it is exact; d == 0 gives 0 (as the model's eps-guarded divisions do
// for an empty event).  The skewness divider saturates (the feature is clipped to [-8, 8) anyway).
//
// Input: data[512] = 2-bit codes, C order of (16 rows, 16 cols, 2 slices); res[14] = pack_14 order
// [x, d0, y, d1, cotA, d2, cotB, d3, M21, M31, M32, M41, M42, M43].
#ifndef NNET_SYMMOE_NF_H_
#define NNET_SYMMOE_NF_H_

#include "ap_fixed.h"
#include "ap_int.h"

namespace nnet {

struct symmoe_nf_config {};

namespace symnf {

constexpr int cmax(int a, int b) { return a > b ? a : b; }
constexpr int pos(int a) { return a > 0 ? a : 0; }
// bits needed for an unsigned value v (v >= 0)
constexpr int ubits(long long v) { return v <= 0 ? 1 : (v == 1 ? 1 : 1 + ubits(v >> 1)); }
constexpr int frac_of(int w, int i) { return w - i; }

template <class T> inline ap_int<T::width> raw(const T &v) {
#pragma HLS INLINE
    ap_int<T::width> r;
    r.range(T::width - 1, 0) = v.range(T::width - 1, 0);
    return r;
}

// floor shift by S (S < 0: left shift), into WO bits (caller guarantees the value fits)
template <int S, int WO, int WI> inline ap_int<WO> shr(const ap_int<WI> &x) {
#pragma HLS INLINE
    const int SR = S > 0 ? S : 0;
    const int SL = S < 0 ? -S : 0;
    ap_int<cmax(WI, WO)> t = x;
    t = t >> SR;
    ap_int<WO> r = t;
    r = r << SL;
    return r;
}

template <int WO, int WI> inline ap_int<WO> sat(const ap_int<WI> &x) {
#pragma HLS INLINE
    const long long hi = (1LL << (WO - 1)) - 1, lo = -(1LL << (WO - 1));
    ap_int<WO> r;
    if (x > ap_int<cmax(WI, WO) + 1>(hi))
        r = hi;
    else if (x < ap_int<cmax(WI, WO) + 1>(lo))
        r = lo;
    else
        r = x;
    return r;
}

template <int WO, int WI> inline ap_int<WO> satu(const ap_int<WI> &x) { // clip to [0, 2^WO - 1], signed holder
#pragma HLS INLINE
    const long long hi = (1LL << WO) - 1;
    ap_int<WO + 1> r;
    if (x > ap_int<cmax(WI, WO + 1) + 1>(hi))
        r = hi;
    else if (x < 0)
        r = 0;
    else
        r = x;
    return r;
}

// floor(n / d) for 0 <= n < d * 2^QB; 0 when d == 0
template <int QB, int WN, int WD> inline ap_uint<QB> udivq(const ap_uint<WN> &n, const ap_uint<WD> &d) {
#pragma HLS INLINE
    typedef ap_uint<cmax(WN, WD + QB)> wt;
    wt r = n;
    ap_uint<QB> q = 0;
    for (int b = QB - 1; b >= 0; b--) {
#pragma HLS UNROLL
        wt t = wt(d) << b;
        if (r >= t) {
            r -= t;
            q[b] = 1;
        }
    }
    if (d == 0)
        q = 0;
    return q;
}

// min(floor(n / d), 2^QB) for any n >= 0; 0 when d == 0
template <int QB, int WN, int WD> inline ap_uint<QB + 1> udivq_sat(const ap_uint<WN> &n, const ap_uint<WD> &d) {
#pragma HLS INLINE
    typedef ap_uint<cmax(WN, WD + QB) + 1> wt;
    ap_uint<QB + 1> q;
    if (d != 0 && wt(n) >= (wt(d) << QB))
        q = ap_uint<QB + 1>(1) << QB;
    else
        q = udivq<QB, WN, WD>(ap_uint<WN>(n), d);
    return q;
}

// sign(n) * floor(|n| * 2^S / d), |result| < 2^QB guaranteed by the caller's bound
template <int QB, int S, int WN, int WD> inline ap_int<QB + 1> tdivs(const ap_int<WN> &n, const ap_uint<WD> &d) {
#pragma HLS INLINE
    ap_uint<WN> a = (n < 0) ? ap_uint<WN>(-n) : ap_uint<WN>(n);
    ap_uint<WN + S> as = ap_uint<WN + S>(a) << S;
    ap_uint<QB> q = udivq<QB, WN + S, WD>(as, d);
    ap_int<QB + 1> r = (n < 0) ? ap_int<QB + 1>(-q) : ap_int<QB + 1>(q);
    return r;
}

// floor(sqrt(v))
template <int WI> inline ap_uint<(WI + 1) / 2> isqrt(const ap_uint<WI> &v) {
#pragma HLS INLINE
    const int WO = (WI + 1) / 2;
    typedef ap_uint<WI + 2> wt;
    wt rem = v;
    ap_uint<WO> res = 0;
    for (int b = WO - 1; b >= 0; b--) {
#pragma HLS UNROLL
        wt trial = (wt(res) << (b + 1)) + (wt(1) << (2 * b));
        if (rem >= trial) {
            rem -= trial;
            res[b] = 1;
        }
    }
    return res;
}

// a + floor((b - a) * r / 2^RB) with a = tab[i], b = tab[i + 1] (tables are increasing)
template <int RB, class tab_T, int WR> inline ap_uint<tab_T::width> interp(const tab_T tab[], int i, const ap_uint<WR> &r) {
#pragma HLS INLINE
    ap_uint<tab_T::width> a = tab[i], b = tab[i + 1];
    ap_uint<tab_T::width> dd = b - a;
    ap_uint<tab_T::width + RB> p = dd * r;
    return a + ap_uint<tab_T::width>(p >> RB);
}

// tanh(x / 2^FIN) at FS frac: odd; table step 2^-TT on [0, TR), linear interpolation, saturates to tab[TR<<TT]
template <class CONFIG_T, int FIN, int WIN>
inline ap_int<CONFIG_T::FS + 2> tanh_fx(const ap_int<WIN> &x, const typename CONFIG_T::tanh_t tab[]) {
#pragma HLS INLINE
    const int RB = FIN - CONFIG_T::TT;
    const int NMAX = CONFIG_T::TR << CONFIG_T::TT;
    ap_uint<WIN> ax = (x < 0) ? ap_uint<WIN>(-x) : ap_uint<WIN>(x);
    ap_uint<WIN> idx = ax >> RB;
    ap_uint<RB> r = ax;
    ap_uint<CONFIG_T::tanh_t::width> v;
    if (idx >= NMAX)
        v = tab[NMAX];
    else
        v = interp<RB>(tab, (int)idx, r);
    ap_int<CONFIG_T::FS + 2> s = (x < 0) ? ap_int<CONFIG_T::FS + 2>(-v) : ap_int<CONFIG_T::FS + 2>(v);
    return s;
}

} // namespace symnf

template <class data_T, class res_T, typename CONFIG_T>
void symmoe_nf(const data_T data[512], res_T res[14], typename CONFIG_T::w_r1_t w_r1[36 * 16],
               typename CONFIG_T::b_r1_t b_r1[16], typename CONFIG_T::w_r2_t w_r2[16 * 4], typename CONFIG_T::b_r2_t b_r2[4],
               typename CONFIG_T::a_sx_t a_sx[4], typename CONFIG_T::a_bx_t a_bx[4], typename CONFIG_T::a_sy_t a_sy[4],
               typename CONFIG_T::a_by_t a_by[4], typename CONFIG_T::a_ka_t a_ka[4], typename CONFIG_T::a_ca_t a_ca[4],
               typename CONFIG_T::a_kb_t a_kb[4], typename CONFIG_T::a_cb_t a_cb[4], typename CONFIG_T::a_sa_t a_sa[4],
               typename CONFIG_T::a_ba_t a_ba[4], typename CONFIG_T::a_sb_t a_sb[4], typename CONFIG_T::a_lb_t a_lb[4],
               typename CONFIG_T::a_bb_t a_bb[4], typename CONFIG_T::w_e1_t w_e1[32 * 8], typename CONFIG_T::b_e1_t b_e1[8],
               typename CONFIG_T::w_e2_t w_e2[8 * 10], typename CONFIG_T::b_e2_t b_e2[10],
               const typename CONFIG_T::ln_t ln_tab[(1 << CONFIG_T::LT) + 1],
               const typename CONFIG_T::exp2_t exp2_tab[(1 << CONFIG_T::TE) + 1],
               const typename CONFIG_T::tanh_t tanh_tab[(CONFIG_T::TR << CONFIG_T::TT) + 1]) {
#pragma HLS INLINE recursive
    // constant tables as logic (muxes of constants), not memories: no BRAM / $readmemh ROMs in the ASIC RTL
#pragma HLS ARRAY_PARTITION variable = ln_tab complete
#pragma HLS ARRAY_PARTITION variable = exp2_tab complete
#pragma HLS ARRAY_PARTITION variable = tanh_tab complete
    using namespace symnf;
    const int FR = CONFIG_T::FR, FM = CONFIG_T::FM, FH = CONFIG_T::FH, IH = CONFIG_T::IH, FZ = CONFIG_T::FZ,
              IZ = CONFIG_T::IZ, FU = CONFIG_T::FU, TE = CONFIG_T::TE, FE = CONFIG_T::FE, FW = CONFIG_T::FW,
              FT = CONFIG_T::FT, FA = CONFIG_T::FA, IA = CONFIG_T::IA, FS = CONFIG_T::FS, FMU = CONFIG_T::FMU,
              FO = CONFIG_T::FO, IO = CONFIG_T::IO, FEH = CONFIG_T::FEH, IEH = CONFIG_T::IEH, LT = CONFIG_T::LT,
              FL = CONFIG_T::FL;
    // port formats (width, frac)
#define SYMNF_W(p) (CONFIG_T::p##_t::width)
#define SYMNF_F(p) (CONFIG_T::p##_t::width - CONFIG_T::p##_t::iwidth)

    // ---------------------------------------------------------------- pixel statistics
    ap_uint<2> c0[16][16], c1[16][16];
    bool m[16][16];
#pragma HLS ARRAY_PARTITION variable = c0 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = c1 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = m complete dim = 0
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            ap_uint<2> a = data[(i * 16 + j) * 2].range(1, 0);
            ap_uint<2> b = data[(i * 16 + j) * 2 + 1].range(1, 0);
            c0[i][j] = a;
            c1[i][j] = b;
            m[i][j] = (ap_uint<3>(a) + ap_uint<3>(b)) >= 2;
        }
    ap_uint<7> Prow[16], Pcol[16];
    ap_uint<6> p0[16], p1[16], gc0[16], gc1[16], hr0[16], hr1[16];
    bool mcol[16], mrow[16];
#pragma HLS ARRAY_PARTITION variable = Prow complete
#pragma HLS ARRAY_PARTITION variable = Pcol complete
#pragma HLS ARRAY_PARTITION variable = p0 complete
#pragma HLS ARRAY_PARTITION variable = p1 complete
#pragma HLS ARRAY_PARTITION variable = gc0 complete
#pragma HLS ARRAY_PARTITION variable = gc1 complete
#pragma HLS ARRAY_PARTITION variable = hr0 complete
#pragma HLS ARRAY_PARTITION variable = hr1 complete
#pragma HLS ARRAY_PARTITION variable = mcol complete
#pragma HLS ARRAY_PARTITION variable = mrow complete
    for (int j = 0; j < 16; j++) {
        ap_uint<6> s0 = 0, s1 = 0, g0 = 0, g1 = 0;
        bool any = false;
        for (int i = 0; i < 16; i++) {
            s0 += c0[i][j];
            s1 += c1[i][j];
            if (m[i][j]) {
                g0 += c0[i][j];
                g1 += c1[i][j];
            }
            any = any || m[i][j];
        }
        p0[j] = s0;
        p1[j] = s1;
        Pcol[j] = ap_uint<7>(s0) + ap_uint<7>(s1);
        gc0[j] = g0;
        gc1[j] = g1;
        mcol[j] = any;
    }
    for (int i = 0; i < 16; i++) {
        ap_uint<6> s0 = 0, s1 = 0, h0 = 0, h1 = 0;
        bool any = false;
        for (int j = 0; j < 16; j++) {
            s0 += c0[i][j];
            s1 += c1[i][j];
            if (m[i][j]) {
                h0 += c0[i][j];
                h1 += c1[i][j];
            }
            any = any || m[i][j];
        }
        Prow[i] = ap_uint<7>(s0) + ap_uint<7>(s1);
        hr0[i] = h0;
        hr1[i] = h1;
        mrow[i] = any;
    }
    ap_uint<11> tot = 0;
    ap_uint<10> T0 = 0, T1 = 0, D0 = 0, D1 = 0;
    ap_int<16> S1 = 0, R1 = 0;
    ap_uint<19> S2 = 0;
    ap_int<24> S3 = 0;
    ap_int<15> Q0 = 0, Q1 = 0, G0 = 0, G1 = 0, H0 = 0, H1 = 0;
    ap_uint<5> wx = 0, wy = 0;
    for (int j = 0; j < 16; j++) {
        const int o = 2 * j - 15; // 2*(j - 7.5)
        tot += Pcol[j];
        T0 += p0[j];
        T1 += p1[j];
        D0 += gc0[j];
        D1 += gc1[j];
        S1 += ap_int<16>(Pcol[j]) * o;
        S2 += ap_uint<19>(Pcol[j]) * (o * o);
        S3 += ap_int<24>(Pcol[j]) * (o * o * o);
        R1 += ap_int<16>(Prow[j]) * o;
        Q0 += ap_int<15>(p0[j]) * o;
        Q1 += ap_int<15>(p1[j]) * o;
        G0 += ap_int<15>(gc0[j]) * o;
        G1 += ap_int<15>(gc1[j]) * o;
        H0 += ap_int<15>(hr0[j]) * o;
        H1 += ap_int<15>(hr1[j]) * o;
        wx += mcol[j] ? 1 : 0;
        wy += mrow[j] ? 1 : 0;
    }

    // ---------------------------------------------------------------- router features (FR frac, range [-8, 8))
    const int WF = FR + 4; // signed feature width
    ap_uint<FR + 1> fu[32]; // pxn (row-indexed), pyn (col-indexed): Prow, Pcol <= tot -> [0, 2^FR]
#pragma HLS ARRAY_PARTITION variable = fu complete
    for (int i = 0; i < 16; i++) {
        fu[i] = udivq<FR + 1, 7 + FR, 11>(ap_uint<7 + FR>(Prow[i]) << FR, tot);
        fu[16 + i] = udivq<FR + 1, 7 + FR, 11>(ap_uint<7 + FR>(Pcol[i]) << FR, tot);
    }
    // moments of the col-indexed profile in pixel units: |S1| <= 15 tot, S2 <= 225 tot, |S3| <= 3375 tot
    ap_int<FM + 4> A1 = tdivs<FM + 3, FM - 1, 16, 11>(S1, tot);
    ap_uint<FM + 6> A2 = udivq<FM + 6, 19 + FM - 2, 11>(ap_uint<19 + FM - 2>(S2) << (FM - 2), tot);
    ap_int<FM + 10> A3 = tdivs<FM + 9, FM - 3, 24, 11>(S3, tot);
    ap_int<2 * FM + 8> a1sq = A1 * A1;
    ap_uint<FM + 6> sq = a1sq >> FM; // <= 56.25 * 2^FM
    ap_int<FM + 7> m2 = ap_int<FM + 7>(A2) - ap_int<FM + 7>(sq);
    ap_int<2 * FM + 12> t3 = ap_int<FM + 6>(A1 * 3) * ap_int<FM + 7>(A2);
    ap_int<2 * FM + 12> t2 = ap_int<FM + 8>(sq * 2) * A1;
    ap_int<FM + 13> m3 = ap_int<FM + 13>(A3) - ap_int<FM + 13>(t3 >> FM) + ap_int<FM + 13>(t2 >> FM);
    ap_uint<FM + 7> vv = (m2 > 0 ? ap_uint<FM + 7>(m2) : ap_uint<FM + 7>(0)) + (ap_uint<FM + 7>(1) << FM);
    ap_uint<FM + 4> sd = isqrt<2 * FM + 7>(ap_uint<2 * FM + 7>(vv) << FM); // FM frac, in [1, 8)
    ap_uint<FM + 7> sd2 = (sd * sd) >> FM;
    ap_uint<FM + 10> sd3 = (sd2 * sd) >> FM; // >= 2^FM
    ap_uint<FM + 13> am3 = m3 < 0 ? ap_uint<FM + 13>(-m3) : ap_uint<FM + 13>(m3);
    ap_uint<FR + 4> qs = udivq_sat<FR + 3, FM + 13 + FR, FM + 10>(ap_uint<FM + 13 + FR>(am3) << FR, sd3);
    ap_int<WF> skew = sat<WF>(m3 < 0 ? ap_int<FR + 5>(-qs) : ap_int<FR + 5>(qs));
    ap_int<FR + 4> cc0 = tdivs<FR + 3, FR - 1, 15, 10>(Q0, T0);
    ap_int<FR + 4> cc1 = tdivs<FR + 3, FR - 1, 15, 10>(Q1, T1);
    ap_int<WF> tasym = sat<WF>(ap_int<FR + 5>(cc1) - ap_int<FR + 5>(cc0));
    // ln(tot + 1) = e ln2 + ln(1 + f/1024), n = 2^e (1 + f/1024)
    ap_uint<11> nn = tot + 1;
    ap_uint<4> e = 0;
    for (int b = 1; b < 11; b++) {
#pragma HLS UNROLL
        if (nn[b])
            e = b;
    }
    ap_uint<21> nsh = ap_uint<21>(nn) << (int)(10 - (int)e);
    ap_uint<10> f = nsh.range(9, 0);
    ap_uint<LT> lidx = f >> (10 - LT);
    ap_uint<10 - LT> lr = f;
    ap_uint<CONFIG_T::ln_t::width> lnm = interp<10 - LT>(ln_tab, (int)lidx, lr);
    ap_int<FL + 5> lsum = ap_int<FL + 5>(ap_uint<FL + 5>(e) * CONFIG_T::LN2_FL) + ap_int<FL + 5>(lnm);
    ap_int<WF> logq = sat<WF>(shr<FL - FR, FL + 5>(lsum));
    ap_int<WF> fstd = sat<WF>(shr<FM - FR, FM + 5>(ap_int<FM + 5>(sd)));
    ap_int<WF> fs[4] = {fstd, skew, tasym, logq};
#pragma HLS ARRAY_PARTITION variable = fs complete

    // ---------------------------------------------------------------- router MLP 36 -> 16 relu -> 4
    const int Fw1 = SYMNF_F(w_r1), Fb1 = SYMNF_F(b_r1), Fw2 = SYMNF_F(w_r2), Fb2 = SYMNF_F(b_r2);
    const int WP1 = cmax(FR + 1 + SYMNF_W(w_r1) + 1, WF + SYMNF_W(w_r1)); // one product
    const int WA1 = cmax(WP1 + 6, SYMNF_W(b_r1) + pos(FR + Fw1 - Fb1)) + 1;
    const int WH = IH + FH;
    ap_uint<WH> h[16];
#pragma HLS ARRAY_PARTITION variable = h complete
    for (int j = 0; j < 16; j++) {
        ap_int<WA1> acc = shr<Fb1 - (FR + Fw1), WA1>(raw(b_r1[j]));
        for (int i = 0; i < 32; i++)
            acc += ap_int<FR + 2>(fu[i]) * raw(w_r1[i * 16 + j]);
        for (int i = 0; i < 4; i++)
            acc += fs[i] * raw(w_r1[(32 + i) * 16 + j]);
        ap_int<WA1> ar = acc > 0 ? acc : ap_int<WA1>(0);
        h[j] = satu<WH>(shr<FR + Fw1 - FH, WA1 + pos(FH - FR - Fw1)>(ap_int<WA1 + pos(FH - FR - Fw1)>(ar)));
    }
    const int WA2 = cmax(WH + 1 + SYMNF_W(w_r2) + 4, SYMNF_W(b_r2) + pos(FH + Fw2 - Fb2)) + 1;
    const int WZ = IZ + FZ;
    ap_int<WZ> z[4];
#pragma HLS ARRAY_PARTITION variable = z complete
    for (int k = 0; k < 4; k++) {
        ap_int<WA2> acc = shr<Fb2 - (FH + Fw2), WA2>(raw(b_r2[k]));
        for (int j = 0; j < 16; j++)
            acc += ap_int<WH + 1>(h[j]) * raw(w_r2[j * 4 + k]);
        z[k] = sat<WZ>(shr<FH + Fw2 - FZ, WA2 + pos(FZ - FH - Fw2)>(ap_int<WA2 + pos(FZ - FH - Fw2)>(acc)));
    }

    // ---------------------------------------------------------------- softmax: e_k = 2^((z_k - zmax) log2 e), w_k = e_k / sum
    ap_int<WZ> zmax = z[0];
    for (int k = 1; k < 4; k++)
        if (z[k] > zmax)
            zmax = z[k];
    ap_uint<FE + 2> ek[4];
#pragma HLS ARRAY_PARTITION variable = ek complete
    ap_uint<FE + 4> se = 0;
    for (int k = 0; k < 4; k++) {
        ap_int<WZ + 1> dz = ap_int<WZ + 1>(z[k]) - ap_int<WZ + 1>(zmax); // <= 0
        ap_int<WZ + 19> pr = dz * ap_int<19>(CONFIG_T::L2E);
        ap_int<WZ + 19> u = pr >> (FZ + 16 - FU);
        ap_int<WZ + 19> ni = u >> FU;
        ap_uint<FU> fr = u.range(FU - 1, 0);
        ap_uint<TE> eidx = fr >> (FU - TE);
        ap_uint<FU - TE> er = fr;
        ap_uint<CONFIG_T::exp2_t::width> mm = interp<FU - TE>(exp2_tab, (int)eidx, er);
        ap_int<WZ + 20> sh = -ni;
        ap_uint<FE + 2> v;
        if (sh > FE + 1)
            v = 0;
        else
            v = mm >> (int)sh;
        ek[k] = v;
        se += v;
    }
    ap_uint<FW + 1> w[4];
#pragma HLS ARRAY_PARTITION variable = w complete
    for (int k = 0; k < 4; k++)
        w[k] = udivq<FW + 1, FE + 2 + FW, FE + 4>(ap_uint<FE + 2 + FW>(ek[k]) << FW, se);

    // ---------------------------------------------------------------- ansatz statistics
    ap_int<FM + 4> B1 = tdivs<FM + 3, FM - 1, 16, 11>(R1, tot);
    ap_int<FT + 5> ux0 = tdivs<FT + 4, FT, 15, 10>(G0, D0), ux1 = tdivs<FT + 4, FT, 15, 10>(G1, D1);
    ap_int<FT + 5> uy0 = tdivs<FT + 4, FT, 15, 10>(H0, D0), uy1 = tdivs<FT + 4, FT, 15, 10>(H1, D1);
    ap_int<FT + 6> dux = ap_int<FT + 6>(ux1) - ap_int<FT + 6>(ux0);
    ap_int<FT + 6> duy = ap_int<FT + 6>(uy1) - ap_int<FT + 6>(uy0);
    ap_uint<4> wx1 = wx > 0 ? ap_uint<4>(wx - 1) : ap_uint<4>(0);
    ap_uint<4> wy1 = wy > 0 ? ap_uint<4>(wy - 1) : ap_uint<4>(0);

    const int Fsx = SYMNF_F(a_sx), Fbx = SYMNF_F(a_bx), Fsy = SYMNF_F(a_sy), Fby = SYMNF_F(a_by), Fka = SYMNF_F(a_ka),
              Fca = SYMNF_F(a_ca), Fkb = SYMNF_F(a_kb), Fcb = SYMNF_F(a_cb), Fsa = SYMNF_F(a_sa), Fba = SYMNF_F(a_ba),
              Fsb = SYMNF_F(a_sb), Flb = SYMNF_F(a_lb), Fbb = SYMNF_F(a_bb);
    // worst-case widths of the expert terms (any port contents)
    const int WMX = cmax(SYMNF_W(a_sx) + FM + 4 - pos(Fsx + FM - FMU) + pos(FMU - Fsx - FM),
                         SYMNF_W(a_bx) + pos(FMU - Fbx)) + 1;
    const int WMY = cmax(SYMNF_W(a_sy) + FM + 4 - pos(Fsy + FM - FMU) + pos(FMU - Fsy - FM),
                         SYMNF_W(a_by) + pos(FMU - Fby)) + 1;
    const int WAG = cmax(cmax(SYMNF_W(a_ca) + pos(FA - Fca), SYMNF_W(a_ka) + FT + 6 + pos(FA - Fka - FT)),
                         cmax(SYMNF_W(a_cb) + pos(FA - Fcb), SYMNF_W(a_kb) + FT + 6 + pos(FA - Fkb - FT))) + 1;
    const int WGA = SYMNF_W(a_sa) + 4;                                        // maga
    const int WGB = cmax(SYMNF_W(a_sb) + 4, SYMNF_W(a_lb) + pos(Fsb - Flb)) + 1; // magb
    const int WCA = cmax(WGA + FS + 2 + pos(FMU - Fsa - FS), SYMNF_W(a_ba) + pos(FMU - Fba)) + 1;
    const int WCB = cmax(WGB + FS + 2 + pos(FMU - Fsb - FS), SYMNF_W(a_bb) + pos(FMU - Fbb)) + 1;
    const int WMU = cmax(cmax(WMX, WMY), cmax(WCA, WCB));
    const int WAA = IA + FA;
    ap_int<WMU> mu[4][4];
#pragma HLS ARRAY_PARTITION variable = mu complete dim = 0
    for (int k = 0; k < 4; k++) {
        ap_int<WMX> mx = ap_int<WMX>(shr<Fsx + FM - FMU, WMX>(ap_int<SYMNF_W(a_sx) + FM + 4>(raw(a_sx[k]) * A1))) +
                         ap_int<WMX>(shr<Fbx - FMU, WMX>(raw(a_bx[k])));
        ap_int<WMY> my = ap_int<WMY>(shr<Fsy + FM - FMU, WMY>(ap_int<SYMNF_W(a_sy) + FM + 4>(raw(a_sy[k]) * B1))) +
                         ap_int<WMY>(shr<Fby - FMU, WMY>(raw(a_by[k])));
        ap_int<WAG> ga = ap_int<WAG>(shr<Fca - FA, WAG>(raw(a_ca[k]))) -
                         ap_int<WAG>(shr<Fka + FT - FA, WAG>(ap_int<SYMNF_W(a_ka) + FT + 6>(raw(a_ka[k]) * dux)));
        ap_int<WAG> gb = ap_int<WAG>(shr<Fcb - FA, WAG>(raw(a_cb[k]))) -
                         ap_int<WAG>(shr<Fkb + FT - FA, WAG>(ap_int<SYMNF_W(a_kb) + FT + 6>(raw(a_kb[k]) * duy)));
        ap_int<WAA> arga = sat<WAA>(ga), argb = sat<WAA>(gb);
        ap_int<FS + 2> ta = tanh_fx<CONFIG_T, FA, WAA>(arga, tanh_tab);
        ap_int<FS + 2> tb = tanh_fx<CONFIG_T, FA, WAA>(argb, tanh_tab);
        ap_int<WGA> maga = ap_int<5>(wx1) * raw(a_sa[k]);
        ap_int<WGB> magb = ap_int<WGB>(ap_int<5>(wy1) * raw(a_sb[k])) + ap_int<WGB>(shr<Flb - Fsb, WGB>(raw(a_lb[k])));
        ap_int<WCA> ca = ap_int<WCA>(shr<Fsa + FS - FMU, WCA>(ap_int<WGA + FS + 2>(maga * ta))) +
                         ap_int<WCA>(shr<Fba - FMU, WCA>(raw(a_ba[k])));
        ap_int<WCB> cb = ap_int<WCB>(shr<Fsb + FS - FMU, WCB>(ap_int<WGB + FS + 2>(magb * tb))) +
                         ap_int<WCB>(shr<Fbb - FMU, WCB>(raw(a_bb[k])));
        mu[k][0] = mx;
        mu[k][1] = my;
        mu[k][2] = ca;
        mu[k][3] = cb;
    }
    const int WO = IO + FO;
    const int WMM = WMU + FW + 2 + 2;
    ap_int<WO> mean[4];
#pragma HLS ARRAY_PARTITION variable = mean complete
    for (int d = 0; d < 4; d++) {
        ap_int<WMM> s = 0;
        for (int k = 0; k < 4; k++)
            s += ap_int<FW + 2>(w[k]) * mu[k][d];
        mean[d] = sat<WO>(shr<FW + FMU - FO, WMM + pos(FO - FW - FMU)>(ap_int<WMM + pos(FO - FW - FMU)>(s)));
    }

    // ---------------------------------------------------------------- error MLP: profiles/32 -> 8 tanh -> 10
    const int Fe1 = SYMNF_F(w_e1), Fbe1 = SYMNF_F(b_e1), Fe2 = SYMNF_F(w_e2), Fbe2 = SYMNF_F(b_e2);
    const int WA3 = cmax(8 + SYMNF_W(w_e1) + 5, SYMNF_W(b_e1) + pos(5 + Fe1 - Fbe1)) + 1;
    const int WEH = IEH + FEH;
    ap_int<FS + 2> th[8];
#pragma HLS ARRAY_PARTITION variable = th complete
    for (int j = 0; j < 8; j++) {
        ap_int<WA3> acc = shr<Fbe1 - (5 + Fe1), WA3>(raw(b_e1[j]));
        for (int i = 0; i < 16; i++) {
            acc += ap_int<8>(Prow[i]) * raw(w_e1[i * 8 + j]);
            acc += ap_int<8>(Pcol[i]) * raw(w_e1[(16 + i) * 8 + j]);
        }
        ap_int<WEH> eh = sat<WEH>(shr<5 + Fe1 - FEH, WA3 + pos(FEH - 5 - Fe1)>(ap_int<WA3 + pos(FEH - 5 - Fe1)>(acc)));
        th[j] = tanh_fx<CONFIG_T, FEH, WEH>(eh, tanh_tab);
    }
    const int WA4 = cmax(FS + 2 + SYMNF_W(w_e2) + 3, SYMNF_W(b_e2) + pos(FS + Fe2 - Fbe2)) + 1;
    ap_int<WO> chol[10];
#pragma HLS ARRAY_PARTITION variable = chol complete
    for (int k = 0; k < 10; k++) {
        ap_int<WA4> acc = shr<Fbe2 - (FS + Fe2), WA4>(raw(b_e2[k]));
        for (int j = 0; j < 8; j++)
            acc += th[j] * raw(w_e2[j * 10 + k]);
        chol[k] = sat<WO>(shr<FS + Fe2 - FO, WA4 + pos(FO - FS - Fe2)>(ap_int<WA4 + pos(FO - FS - Fe2)>(acc)));
    }

    // ---------------------------------------------------------------- pack_14
    ap_int<WO> o[14] = {mean[0], chol[0], mean[1], chol[1], mean[2], chol[2], mean[3], chol[3],
                        chol[4], chol[5], chol[6], chol[7], chol[8], chol[9]};
#pragma HLS ARRAY_PARTITION variable = o complete
    for (int i = 0; i < 14; i++) {
        res_T r;
        r.range(WO - 1, 0) = o[i].range(WO - 1, 0);
        res[i] = r;
    }
#undef SYMNF_W
#undef SYMNF_F
}

} // namespace nnet

#endif
