"""Frozen-encoder evaluation, V-JEPA protocol: official AttentiveClassifier probe on the
frozen TARGET encoder, 20 epochs, AdamW lr 1e-3 (constant, no warmup), wd 0.01 (their
k400 eval config). Head = 14 Gaussian parameters, trained with our v2 NLL (torch port,
checked against TF). Reported: val plain NLL in the ledger convention -- the NLL summed
over all 37,919 val events / 8 batches -- so it is comparable with -40,162 etc.

usage: probe_attentive.py --ckpt <last.pt | random> --out <dir>
"""
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain_vjepa import build, NF, DATA, JEPA
sys.path.insert(0, JEPA)
from src.models.attentive_pooler import AttentiveClassifier
from nll_torch import custom_loss_v2
dev = torch.device('cuda')
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=0.01)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--limit', type=int, default=0, help='smoke test only')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
ns = json.load(open(os.path.join(DATA, 'norm.json')))
def load(tag):
    x = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(DATA, f'{tag}_x.npy'), mmap_mode='r')[..., :NF])).pin_memory()
    return x, torch.from_numpy(np.load(os.path.join(DATA, f'{tag}_y.npy')))
Xtr, Ytr = load('train'); Xva, Yva = load('val')
if a.limit: Xtr, Ytr, Xva, Yva = Xtr[:a.limit], Ytr[:a.limit], Xva[:a.limit], Yva[:a.limit]
enc, _ = build()
if a.ckpt != 'random':
    s = torch.load(a.ckpt, map_location='cpu', weights_only=False); enc.load_state_dict(s['tgt']); ep = s['epoch']
else: ep = 0
enc.to(dev).eval()
for p in enc.parameters(): p.requires_grad = False
probe = AttentiveClassifier(embed_dim=enc.backbone.embed_dim, num_heads=enc.backbone.num_heads,
                            depth=1, num_classes=14).to(dev)
opt = torch.optim.AdamW(probe.parameters(), lr=a.lr, weight_decay=a.wd)
def feats(xb):
    xb = xb.to(dev, non_blocking=True)
    c = ((xb.float() - ns['mean']) / ns['std']).permute(0, 3, 1, 2).unsqueeze(1)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return enc(c).float()
# The encoder is FROZEN, so its features are identical every probe epoch: compute them
# once (same bf16 forward as before) into pinned host RAM (~23 GB train, ~6 GB val)
# instead of re-running the encoder over 152k events 20 times. Results unchanged.
def cache(X):
    F_ = torch.empty((len(X), 400, enc.backbone.embed_dim), dtype=torch.float16).pin_memory()
    for i in range(0, len(X), 512): F_[i:i+512] = feats(X[i:i+512]).half().cpu()
    return F_
_t = time.time(); Ftr = cache(Xtr); Fva = cache(Xva); del Xtr, Xva
print(f'cached frozen features in {time.time()-_t:.0f}s', flush=True)
def evaluate():
    probe.eval(); tot = 0.0
    with torch.no_grad():
        for i in range(0, len(Fva), 1000):
            tot += float(custom_loss_v2(Yva[i:i+1000].to(dev), probe(Fva[i:i+1000].to(dev).float())))
    probe.train(); return tot / 8.0                    # ledger convention: sum over val / 8 batches
log = []
for e in range(a.epochs):
    t0 = time.time(); perm = torch.randperm(len(Ftr)); tl = 0.0
    for b in range(0, len(Ftr) - a.batch + 1, a.batch):
        idx = perm[b:b + a.batch]
        loss = custom_loss_v2(Ytr[idx].to(dev), probe(Ftr[idx].to(dev, non_blocking=True).float()))
        opt.zero_grad(set_to_none=True); (loss / a.batch).backward(); opt.step(); tl += float(loss)
    v = evaluate(); log.append({'epoch': e + 1, 'train_nll_per_event': tl / len(Ftr), 'val_plain_nll': v})
    print(f'probe ep {e+1}/{a.epochs} train/event {tl/len(Ftr):.3f}  val_plain_nll {v:,.0f}  {time.time()-t0:.0f}s', flush=True)
best = min(r['val_plain_nll'] for r in log)
json.dump({'ckpt': a.ckpt, 'pretrain_epoch': ep, 'best_val_plain_nll': best, 'final_val_plain_nll': log[-1]['val_plain_nll'],
           'log': log, 'protocol': vars(a)}, open(os.path.join(a.out, 'probe.json'), 'w'), indent=1)
print(f'BEST val_plain_nll {best:,.0f} (pretrain epoch {ep})', flush=True)
