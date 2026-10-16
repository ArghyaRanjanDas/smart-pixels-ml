"""End-to-end fine-tuning of a pretrained V-JEPA encoder on the track regression (O26 ceiling test).

The standard step after self-supervised pretraining (VideoMAE / V-JEPA fine-tuning
practice): encoder initialised from the pretrained TARGET encoder, an attentive head
(the official AttentiveClassifier, as in our probe) to 14 Gaussian parameters, and the
WHOLE network trained on the plain v2 NLL (torch port, checked against TF).
AdamW, layer-wise lr decay 0.75 over the 12 blocks, weight decay 0.05, warmup then
cosine, bf16, grad clip 10. best.pt is picked on val plain NLL in the ledger
convention (sum over the 37,919 val events / 8), so it compares directly with the
two-slice ViT_D's -40,162. --accum splits each batch for memory (same gradient).
"""
import os, sys, json, time, math, argparse, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain_vjepa import build, NF, DATA, JEPA
sys.path.insert(0, JEPA)
from src.models.attentive_pooler import AttentiveClassifier
from nll_torch import custom_loss_v2
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--warmup', type=int, default=10)
ap.add_argument('--batch', type=int, default=512); ap.add_argument('--accum', type=int, default=1)
ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--min-lr', type=float, default=1e-6)
ap.add_argument('--wd', type=float, default=0.05); ap.add_argument('--layer-decay', type=float, default=0.75)
ap.add_argument('--clip', type=float, default=10.0); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max-iters', type=int, default=0); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device('cuda'); torch.set_num_threads(8)
ns = json.load(open(os.path.join(DATA, 'norm.json')))
def load(tag):
    x = np.load(os.path.join(DATA, f'{tag}_x.npy'), mmap_mode='r'); y = np.load(os.path.join(DATA, f'{tag}_y.npy'))
    if a.limit: x, y = x[:a.limit], y[:a.limit]
    return torch.from_numpy(np.ascontiguousarray(x[..., :NF])).pin_memory(), torch.from_numpy(y)
Xtr, Ytr = load('train'); Xva, Yva = load('val')

enc, _ = build(); s = torch.load(a.ckpt, map_location='cpu', weights_only=False)
enc.load_state_dict(s['tgt']); enc.to(dev)
head = AttentiveClassifier(embed_dim=enc.backbone.embed_dim, num_heads=enc.backbone.num_heads,
                           depth=1, num_classes=14).to(dev)
nb = len(enc.backbone.blocks)
def layer_id(n):        # 0 = embeddings, 1..nb = blocks, nb+1 = final norm + head
    if 'blocks.' in n: return int(n.split('blocks.')[1].split('.')[0]) + 1
    if 'patch_embed' in n or 'pos_embed' in n: return 0
    return nb + 1
groups = {}
for mod, pre in ((enc, ''), (head, 'head.')):
    for n, p in mod.named_parameters():
        lid = layer_id(pre + n) if mod is enc else nb + 1
        nowd = p.ndim == 1 or n.endswith('.bias') or 'pos_embed' in n
        k = (lid, nowd)
        groups.setdefault(k, {'params': [], 'weight_decay': 0.0 if nowd else a.wd,
                              'lr_scale': a.layer_decay ** (nb + 1 - lid)})['params'].append(p)
opt = torch.optim.AdamW(list(groups.values()), lr=a.lr, betas=(0.9, 0.999))
ipe = len(Xtr) // a.batch; total = ipe * a.epochs; wsteps = ipe * a.warmup
def lr_at(it):
    if it < wsteps: return a.lr * it / max(1, wsteps)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * (it - wsteps) / max(1, total - wsteps)))
def clips_of(xb):
    xb = xb.to(dev, non_blocking=True)
    return ((xb.float() - ns['mean']) / ns['std']).permute(0, 3, 1, 2).unsqueeze(1)
def evaluate():
    enc.eval(); head.eval(); tot = 0.0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(Xva), 500):
            tot += float(custom_loss_v2(Yva[i:i+500].to(dev), head(enc(clips_of(Xva[i:i+500]))).float()))
    enc.train(); head.train(); return tot / 8.0
ck = os.path.join(a.out, 'last.pt'); ep0 = it = 0; best = float('inf')
if os.path.exists(ck):
    st = torch.load(ck, map_location=dev, weights_only=False)
    enc.load_state_dict(st['enc']); head.load_state_dict(st['head']); opt.load_state_dict(st['opt'])
    ep0, it, best = st['epoch'], st['it'], st['best']
json.dump({**vars(a), 'pretrain_epoch': s['epoch'], 'ipe': ipe,
           'params': sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())},
          open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
log = open(os.path.join(a.out, 'finetune_log.csv'), 'a', newline=''); w = csv.writer(log)
if ep0 == 0: w.writerow(['epoch', 'it', 'train_nll_per_event', 'val_plain_nll', 'lr', 'sec'])
enc.train(); head.train()
for ep in range(ep0, a.epochs):
    t0 = time.time(); perm = torch.randperm(len(Xtr)); tl = []
    for b in range(ipe):
        lr = lr_at(it)
        for g in opt.param_groups: g['lr'] = lr * g['lr_scale']
        idx = perm[b * a.batch:(b + 1) * a.batch]; mb = a.batch // a.accum
        opt.zero_grad(set_to_none=True)
        for k in range(a.accum):
            sl = idx[k * mb:(k + 1) * mb]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = head(enc(clips_of(Xtr[sl])))
            loss = custom_loss_v2(Ytr[sl].to(dev), p.float())
            (loss / a.batch).backward(); tl.append(loss.detach())
        if a.clip:
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), a.clip)
        opt.step(); it += 1
        if a.max_iters and it >= a.max_iters: break
    v = evaluate(); tr = float(torch.stack(tl).sum()) / (len(tl) * mb)
    if v < best:
        best = v; torch.save({'enc': enc.state_dict(), 'head': head.state_dict(), 'epoch': ep + 1, 'val': v},
                             os.path.join(a.out, 'best.pt'))
    w.writerow([ep + 1, it, tr, v, lr, round(time.time() - t0, 1)]); log.flush()
    print(f'ft ep {ep+1}/{a.epochs} train/event {tr:.3f} val_plain_nll {v:,.0f} best {best:,.0f} lr {lr:.2e} '
          f'{time.time()-t0:.0f}s peak_gpu {torch.cuda.max_memory_allocated()/1e9:.1f}GB', flush=True)
    torch.save({'enc': enc.state_dict(), 'head': head.state_dict(), 'opt': opt.state_dict(),
                'epoch': ep + 1, 'it': it, 'best': best}, ck + '.tmp'); os.replace(ck + '.tmp', ck)
    if a.max_iters and it >= a.max_iters: break
print(f'BEST val_plain_nll {best:,.0f}', flush=True)
