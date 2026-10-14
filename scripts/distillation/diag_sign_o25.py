"""Do the wide float+unconstrained seeds know the angle's magnitude but not its sign?

Their angle pulls are a double spike at exactly +-1, not a wide Gaussian. A spike
at |pull|=1 means |true - pred| ~ sigma event by event. This measures it directly:
  sign(pred)=sign(true)    -- does the prediction carry the sign?
  corr(sigma, |true|)      -- is the magnitude hidden in the uncertainty?
  frac |pull| in [0.8,1.2] -- a unit Gaussian gives 19.4%
Output kept beside the figures as runs/perf_plots_o25/bimodal_sign_diag.txt.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_overlay_o25 as O
print(f"{'seed':8s} {'sign(pred)=sign(true)':>22s} {'corr(sig,|true-pred|)':>22s} {'corr(sig,|true|)':>17s} {'frac |pull| in [0.8,1.2]':>25s}")
for sd in ('50842','50242','50342'):
    df,_ = O.pred('runs/o25_float_plain','f',5,sd)
    for v in ('cotA','cotB'):
        t=df[v+'true'].values; p=df[v].values; s=df['sigma'+v].values
        sign = np.mean(np.sign(p)==np.sign(t))
        c_res = np.corrcoef(s, np.abs(t-p))[0,1]
        c_abs = np.corrcoef(s, np.abs(t))[0,1]
        pull  = np.abs((t-p)/s)
        spike = np.mean((pull>0.8)&(pull<1.2))
        print(f"s{sd} {v:4s} {sign:18.1%}     {c_res:18.3f}     {c_abs:14.3f}     {spike:20.1%}")
