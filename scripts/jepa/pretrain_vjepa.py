"""V-JEPA self-supervised pretraining on the 101-slice smart-pixel clusters (O26).

Sticks to the published recipe: the encoder, predictor, masking collators,
apply_masks and the AdamW / warmup-cosine / cosine-WD schedulers are IMPORTED
from the official facebookresearch/jepa code (CC BY-NC 4.0, not copied into this
repo), and train_step() mirrors app/vjepa/train.py:
  target = LayerNorm(EMA-encoder(full clip)) at the masked tokens (stop-grad)
  pred   = predictor(encoder(visible tokens), mask tokens)
  loss   = L1(pred, target), averaged over the mask generators
  EMA momentum linear 0.998 -> 1.0, bf16 autocast, grad clip 10 after warmup.

Deviations forced by the data (16x16 pixels x 101 slices, 1 channel):
  in_chans=1; patch 4x4 px (4x4 spatial grid) and tubelet 4 slices on the first
  100 slices -> 400 tokens; ViT-Tiny encoder; predictor 96 wide x 6 deep.
Masking variants (--mask): multiblock (V-JEPA default) | causal (multi-block with
context limited to the first 75% of time, V-JEPA's best causal ablation) |
randtube (random tube, ratio 0.9) | temporal (multi-block whose blocks span the
whole frame but part of the time axis).
"""
import os, sys, json, time, math, argparse, copy, csv
import numpy as np, torch, torch.nn.functional as F
JEPA = '/work/users/das214/external/jepa'
sys.path.insert(0, JEPA)
import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.masks.utils import apply_masks
from src.utils.tensors import trunc_normal_
from app.vjepa.utils import init_opt
from src.masks.multiblock3d import MaskCollator as MBCollator
from src.masks.random_tube import MaskCollator as TubeCollator

R = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA = os.path.join(R, 'runs/o26_jepa/data')
NF, PS, TS, IMG = 100, 4, 4, 16

def mb(n, s, t=(1.0, 1.0), ar=(0.75, 1.5), keep=1.0):
    return dict(aspect_ratio=ar, num_blocks=n, spatial_scale=s, temporal_scale=t,
                max_temporal_keep=keep, max_keep=None)
MASKS = {   # V-JEPA vitl16.yaml short-range + long-range, and its ablation variants
    'multiblock': [mb(8, (0.15, 0.15)), mb(2, (0.7, 0.7))],
    'causal':     [mb(8, (0.15, 0.15), keep=0.75), mb(2, (0.7, 0.7), keep=0.75)],
    'temporal':   [mb(8, (1.0, 1.0), t=(0.15, 0.15), ar=(1.0, 1.0)),
                   mb(2, (1.0, 1.0), t=(0.7, 0.7), ar=(1.0, 1.0))],
    'randtube':   [dict(ratio=0.9)],
}

def build(pred_dim=96, pred_depth=6):
    enc = MultiMaskWrapper(video_vit.vit_tiny(img_size=IMG, patch_size=PS, num_frames=NF, tubelet_size=TS,
                                              in_chans=1, uniform_power=True, use_sdpa=True))
    pred = PredictorMultiMaskWrapper(vit_pred.vit_predictor(
        img_size=IMG, use_mask_tokens=True, patch_size=PS, num_frames=NF, tubelet_size=TS,
        embed_dim=enc.backbone.embed_dim, predictor_embed_dim=pred_dim, depth=pred_depth,
        num_heads=enc.backbone.num_heads, uniform_power=True, num_mask_tokens=2,
        zero_init_mask_tokens=True, use_sdpa=True))
    def init_weights(m):          # == app/vjepa/utils.init_video_model
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0); torch.nn.init.constant_(m.weight, 1.0)
    for mdl in (enc, pred):
        for m in mdl.modules(): init_weights(m)
    return enc, pred

class Clips(torch.utils.data.Dataset):
    """(16,16,101) float16 memmap -> (1,100,16,16) standardised float32 clip."""
    def __init__(s, x, mu, sd): s.x, s.mu, s.sd = x, mu, sd
    def __len__(s): return len(s.x)
    def __getitem__(s, i):
        a = torch.from_numpy(np.asarray(s.x[i, :, :, :NF], np.float32))      # (16,16,100)
        return ((a.permute(2, 0, 1).unsqueeze(0) - s.mu) / s.sd, 0)           # (1,100,16,16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', required=True, choices=list(MASKS))
    ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--warmup', type=int, default=40)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=6.25e-4); ap.add_argument('--start-lr', type=float, default=2e-4)
    ap.add_argument('--final-lr', type=float, default=1e-6)
    ap.add_argument('--wd', type=float, default=0.04); ap.add_argument('--final-wd', type=float, default=0.4)
    ap.add_argument('--ema', type=float, nargs=2, default=(0.998, 1.0))
    ap.add_argument('--ipe-scale', type=float, default=1.25); ap.add_argument('--clip-grad', type=float, default=10.0)
    ap.add_argument('--max-iters', type=int, default=0, help='smoke test: stop after N iterations')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda')
    # Small CPU ops spread over all 128 threads only spin (measured: 39 cores busy for
    # no gain; the mask collator is 230 ms on 1 thread or 128).
    torch.set_num_threads(8)

    xtr = np.load(os.path.join(DATA, 'train_x.npy'), mmap_mode='r')
    st = os.path.join(DATA, 'norm.json')
    if not os.path.exists(st):     # global standardisation from a fixed train subsample
        sub = np.asarray(xtr[np.random.RandomState(0).choice(len(xtr), 4000, replace=False)][..., :NF], np.float32)
        json.dump({'mean': float(sub.mean()), 'std': float(sub.std())}, open(st, 'w'))
    ns = json.load(open(st))
    # The pod's /dev/shm is 64 MB, smaller than one batch, so DataLoader workers
    # cannot hand batches over. The whole train set (152k x 16x16x100 fp16 = 7.8 GB)
    # sits in pinned host RAM and each batch is gathered there and copied over; the
    # OFFICIAL collator still draws the masks (they depend only on the batch size).
    # (Holding it on the GPU instead cost 7.8 GB per run and two runs OOM'd.)
    X = torch.from_numpy(np.ascontiguousarray(xtr[..., :NF])).pin_memory()      # (N,16,16,100) fp16
    MU, SD = ns['mean'], ns['std']
    coll = (TubeCollator if a.mask == 'randtube' else MBCollator)(
        cfgs_mask=MASKS[a.mask], crop_size=IMG, num_frames=NF, patch_size=PS, tubelet_size=TS)
    _dummy = [(torch.zeros(1), 0)] * a.batch
    ipe = len(X) // a.batch
    # The official collator is ~230 ms of pure-Python work per batch of 512 -- ~40% of
    # an iteration when run inline. Draw the NEXT batch's masks in a background thread
    # while the GPU works on this one (same generator, same distribution; it is what
    # V-JEPA's DataLoader workers do, which this pod's 64 MB /dev/shm rules out).
    from concurrent.futures import ThreadPoolExecutor
    _pool = ThreadPoolExecutor(max_workers=1)
    def batches():
        perm = torch.randperm(len(X))
        nxt = _pool.submit(coll, _dummy)
        for b in range(ipe):
            _, m_enc, m_pred = nxt.result()
            nxt = _pool.submit(coll, _dummy)
            xb = X[perm[b * a.batch:(b + 1) * a.batch]].pin_memory().to(dev, non_blocking=True)
            clips = ((xb.float() - MU) / SD).permute(0, 3, 1, 2).unsqueeze(1)
            yield clips.contiguous(), m_enc, m_pred

    enc, pred = build(); enc.to(dev); pred.to(dev)
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters(): p.requires_grad = False
    opt, scaler, sched, wd_sched = init_opt(enc, pred, iterations_per_epoch=ipe, start_lr=a.start_lr,
        ref_lr=a.lr, warmup=a.warmup, num_epochs=a.epochs, wd=a.wd, final_wd=a.final_wd,
        final_lr=a.final_lr, mixed_precision=True, ipe_scale=a.ipe_scale)
    total = int(ipe * a.epochs * a.ipe_scale)
    mom = lambda it: a.ema[0] + it * (a.ema[1] - a.ema[0]) / total

    ck = os.path.join(a.out, 'last.pt'); ep0 = 0; it = 0
    if os.path.exists(ck):         # exact resume: weights, target, optimiser, schedulers, counters
        s = torch.load(ck, map_location=dev, weights_only=False)
        enc.load_state_dict(s['enc']); pred.load_state_dict(s['pred']); tgt.load_state_dict(s['tgt'])
        opt.load_state_dict(s['opt']); ep0, it = s['epoch'], s['it']
        for _ in range(it): sched.step(); wd_sched.step()
    json.dump({**vars(a), 'ipe': ipe, 'enc_params': sum(p.numel() for p in enc.parameters()),
               'pred_params': sum(p.numel() for p in pred.parameters()), 'norm': ns,
               'masks': MASKS[a.mask], 'tokens': (NF // TS) * (IMG // PS) ** 2},
              open(os.path.join(a.out, 'config.json'), 'w'), indent=1, default=str)
    log = open(os.path.join(a.out, 'pretrain_log.csv'), 'a', newline=''); w = csv.writer(log)
    if ep0 == 0: w.writerow(['epoch', 'it', 'loss', 'lr', 'wd', 'momentum', 'target_std', 'n_ctx', 'n_pred', 'sec'])

    for ep in range(ep0, a.epochs):
        enc.train(); pred.train(); t0 = time.time(); L = []; tstd = []
        for clips, m_enc, m_pred in batches():
            m_enc = [m.to(dev, non_blocking=True) for m in m_enc]
            m_pred = [m.to(dev, non_blocking=True) for m in m_pred]
            lr = sched.step(); wd = wd_sched.step()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    h = tgt(clips); h = F.layer_norm(h, (h.size(-1),))
                    tstd.append(h.float().std(0).mean())             # collapse monitor (stays on GPU)
                    h = apply_masks(h, m_pred, concat=False)
                z = pred(enc(clips, m_enc), h, m_enc, m_pred)
                loss = sum(torch.mean(torch.abs(zi - hi)) for zi, hi in zip(z, h)) / len(m_pred)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            if ep > a.warmup and a.clip_grad:
                torch.nn.utils.clip_grad_norm_(enc.parameters(), a.clip_grad)
                torch.nn.utils.clip_grad_norm_(pred.parameters(), a.clip_grad)
            scaler.step(opt); scaler.update()
            m = mom(it)
            with torch.no_grad():
                for pq, pk in zip(enc.parameters(), tgt.parameters()):
                    pk.data.mul_(m).add_((1. - m) * pq.detach().data)
            L.append(loss.detach()); it += 1                 # no per-iteration host sync
            if a.max_iters and it >= a.max_iters: break
        L = float(torch.stack(L).float().mean()); tstd = float(torch.stack(tstd).float().mean())
        w.writerow([ep + 1, it, L, lr, wd, m, tstd, m_enc[0].shape[1], m_pred[0].shape[1],
                    round(time.time() - t0, 1)]); log.flush()
        print(f'ep {ep+1}/{a.epochs} loss {L:.4f} target_std {tstd:.3f} lr {lr:.2e} '
              f'mom {m:.5f} {time.time()-t0:.0f}s peak_gpu {torch.cuda.max_memory_allocated()/1e9:.1f}GB', flush=True)
        torch.save({'enc': enc.state_dict(), 'pred': pred.state_dict(), 'tgt': tgt.state_dict(),
                    'opt': opt.state_dict(), 'epoch': ep + 1, 'it': it}, ck + '.tmp')
        os.replace(ck + '.tmp', ck)
        if (ep + 1) % 25 == 0: torch.save({'tgt': tgt.state_dict(), 'enc': enc.state_dict(), 'epoch': ep + 1},
                                          os.path.join(a.out, f'ckpt_ep{ep+1:04d}.pt'))
        if a.max_iters and it >= a.max_iters: break

if __name__ == '__main__':
    main()
