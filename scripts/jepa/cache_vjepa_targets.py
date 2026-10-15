"""Cache frozen V-JEPA target-encoder features as QConv2D distillation targets (O26 ASIC arm).

Per event: target encoder on the full analog clip (first 100 slices) -> 400 tokens x 192,
LayerNorm per token exactly as V-JEPA builds its targets, then averaged over the 25
temporal tubelets -> 4x4 spatial grid x 192 = 3072 dims (the QConv2D also ends on a
4x4 grid). Written in the per-batch layout run_o26_latent.py reads (31 train batches
of 5000 with a partial last one, 8 val), checked against the TF generator's labels.
usage: cache_vjepa_targets.py <ckpt> <outdir>
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain_vjepa import build, NF, DATA
ck, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
dev = torch.device('cuda'); torch.set_num_threads(8)
ns = json.load(open(os.path.join(DATA, 'norm.json')))
enc, _ = build(); s = torch.load(ck, map_location='cpu', weights_only=False)
enc.load_state_dict(s['tgt']); enc.to(dev).eval()
# the TF generator's own per-batch labels (O25 KL teacher cache) define the batch layout
ref = {t: np.load(f'runs/o25_qconv2d/_teacher_cache/teacher_{t}.npz') for t in ('train', 'val')}
for tag in ('train', 'val'):
    X = np.load(os.path.join(DATA, f'{tag}_x.npy'), mmap_mode='r'); Y = np.load(os.path.join(DATA, f'{tag}_y.npy'))
    sizes = [ref[tag][f'Y{i}'].shape[0] for i in range(int(ref[tag]['n']))]
    assert sum(sizes) == len(X), (tag, sum(sizes), len(X))
    Z = []; k = 0
    for i, n in enumerate(sizes):
        assert np.array_equal(Y[k:k + n], ref[tag][f'Y{i}']), f'{tag} batch {i}: event order differs'
        zb = []
        for j in range(k, k + n, 500):
            x = torch.from_numpy(np.asarray(X[j:min(j + 500, k + n), :, :, :NF], np.float32)).to(dev)
            c = ((x - ns['mean']) / ns['std']).permute(0, 3, 1, 2).unsqueeze(1)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                h = enc(c).float()
            h = F.layer_norm(h, (h.size(-1),))                       # V-JEPA target normalisation
            h = h.reshape(h.shape[0], 25, 16, h.shape[-1]).mean(1)   # token order (t,h,w): mean over time
            zb.append(h.reshape(h.shape[0], -1).half().cpu().numpy())
        Z.append(np.concatenate(zb)); k += n
    with open(os.path.join(out, f'latent_{tag}.npz'), 'wb') as fh:
        np.savez(fh, n=len(Z), **{f'Z{i}': z for i, z in enumerate(Z)})
    allz = np.concatenate(Z).astype(np.float32)
    print(f'{tag}: {len(Z)} batches, {len(allz)} events, dim {allz.shape[1]}, mean {allz.mean():.3f} std {allz.std():.3f}, labels match', flush=True)
json.dump({'layer': 'vjepa target encoder, LN tokens, mean over time, 4x4x192', 'dim': 3072, 'dead_units': [],
           'teacher': ck, 'pretrain_epoch': s['epoch']}, open(os.path.join(out, 'stats.json'), 'w'), indent=1)
