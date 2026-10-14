"""Collapse diagnostic for a V-JEPA checkpoint vs a random-init encoder.

Per encoder: on 2,000 val events,
  * effective rank (exp of spectral entropy) of event-level features (mean over tokens)
  * cross-event feature std, separately for tokens WITH charge and EMPTY tokens
A collapsed encoder has effective rank -> 1 and signal-token std -> 0. Sparse data
alone lowers the empty-token std but should leave the signal tokens varied.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain_vjepa import build, NF, PS, TS, DATA
dev = torch.device('cuda')
ck = sys.argv[1]
ns = json.load(open(os.path.join(DATA, 'norm.json')))
xv = np.load(os.path.join(DATA, 'val_x.npy'), mmap_mode='r')
x = torch.from_numpy(np.asarray(xv[:2000, :, :, :NF], np.float32))            # (B,16,16,100)
clips = ((x - ns['mean']) / ns['std']).permute(0, 3, 1, 2).unsqueeze(1).to(dev)
# charge per token: |raw| summed over each 4x4 px x 4-slice tubelet, token order = (t, h, w)
q = x.abs().permute(0, 3, 1, 2).reshape(2000, NF // TS, TS, 4, PS, 4, PS).sum((2, 4, 6)).reshape(2000, -1)
sig = q > np.percentile(q.numpy(), 75)          # top-quartile-charge tokens

def stats(enc):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        h = enc(clips).float()
    h = F.layer_norm(h, (h.size(-1),)).cpu()                                   # as the targets are built
    ev = h.mean(1); ev = ev - ev.mean(0)
    s = torch.linalg.svdvals(ev); p = s / s.sum(); er = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    sd_sig = float(h[sig].std(0).mean()) if sig.any() else float('nan')
    sd_emp = float(h[~sig].std(0).mean())
    return er, sd_sig, sd_emp

e0, _ = build(); e0.to(dev).eval()
s = torch.load(ck, map_location=dev, weights_only=False)
e1, _ = build(); e1.load_state_dict(s['tgt']); e1.to(dev).eval()
for name, e in (('random init', e0), (f"target enc ep{s['epoch']}", e1)):
    er, a, b = stats(e)
    print(f'{name:18s} eff.rank {er:6.1f}/192   std signal-tokens {a:.3f}   std empty-tokens {b:.3f}')
