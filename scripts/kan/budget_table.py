"""Resource comparison for the KAN study: every architecture under ONE cost model.

Cost model (programmable ASIC, inference only):
  stored bits  = weights x weight bits (+ biases)       for conv / dense layers
               = edges x 2^(input bits) x entry bits     for KAN layers (no biases: absorbed)
  multiplies   = MACs of conv / dense layers (KAN edges are lookups: 0)
  lookups      = KAN edge evaluations per event
  additions    = MAC adds (conv/dense) or edge sums (KAN) per event
Conv layers run at 14x14 output positions; pooling 3x3 -> 4x4 grid (80 features).
"""
POS = 14 * 14

def conv(name, k, cin, cout, wb, sep=False):
    if sep:   # depthwise kxk per input channel + pointwise cin->cout, bias on output
        w = k * k * cin + cin * cout; macs = POS * (k * k * cin + cin * cout)
    else:
        w = k * k * cin * cout; macs = POS * k * k * cin * cout
    p = w + cout
    return dict(layer=name, kind='conv', params=p, bits=p * wb, mult=macs, lookups=0, adds=macs, desc=f'{wb}-bit weights')

def dense(name, din, dout, wb):
    p = din * dout + dout
    return dict(layer=name, kind='dense', params=p, bits=p * wb, mult=din * dout, lookups=0, adds=din * dout, desc=f'{wb}-bit weights')

def kan(name, din, dout, inb, entb, positions=1):
    edges = din * dout; ent = edges * 2 ** inb
    return dict(layer=name, kind='KAN', params=ent, bits=ent * entb, mult=0, lookups=positions * edges,
                adds=positions * dout * (din - 1), desc=f'{edges} edges x {2**inb} entries x {entb}b')

def total(rows): return {k: sum(r[k] for r in rows) for k in ('params', 'bits', 'mult', 'lookups', 'adds')}

head_mlp = [dense('dense 80->16', 80, 16, 8), dense('dense 16->16', 16, 16, 8), dense('dense 16->14', 16, 14, 8)]
conv_q   = [conv('sep-conv 3x3 2->5', 3, 2, 5, 4, sep=True), conv('conv 1x1 5->5', 1, 5, 5, 4)]
def kconv(entb): return [kan('KAN conv 3x3 (2-bit in)', 18, 5, 2, entb, POS), kan('KAN conv 1x1 (4-bit in)', 5, 5, 4, entb, POS)]
def khead(h, inb, hb, entb): return [kan(f'KAN head 80->{h} ({inb}-bit in)', 80, h, inb, entb), kan(f'KAN head {h}->14 ({hb}-bit in)', h, 14, hb, entb)]

MODELS = [
 ('A', 'QConv2D (baseline)', conv_q + head_mlp),
 ('B1', 'KAN conv 2/4/4 + MLP head  [built]', kconv(4) + head_mlp),
 ('B2', 'KAN conv 2/8/4 + MLP head', kconv(8) + head_mlp),
 ('C1', 'Q-conv + KAN head [80->3->14], 3-bit in, 6-bit tables', conv_q + khead(3, 3, 3, 6)),
 ('C2', 'Q-conv + KAN head [80->2->14], 4-bit in, 4-bit tables', conv_q + khead(2, 4, 4, 4)),
 ('C3', 'Q-conv + KAN head [80->9->14], 2-bit in, 4-bit tables', conv_q + khead(9, 2, 2, 4)),
 ('D1', 'Full KAN: KAN conv 2/4/4 + KAN head [80->7->14] 2-bit', kconv(4) + khead(7, 2, 2, 4)),
 ('D2', 'Full KAN: KAN conv 2/4/4 + KAN head [80->2->14] 3-bit/6b', kconv(4) + khead(2, 3, 3, 6)),
 ('E1', 'Pooled KAN: pool->4x4x2=32, KAN [32->5->14] 4-bit/4b', [kan('KAN 32->5 (4-bit in)', 32, 5, 4, 4), kan('KAN 5->14 (4-bit in)', 5, 14, 4, 4)]),
 ('E2', 'Pooled KAN: pool->4x4x2=32, KAN [32->12->14] 3-bit/4b', [kan('KAN 32->12 (3-bit in)', 32, 12, 3, 4), kan('KAN 12->14 (3-bit in)', 12, 14, 3, 4)]),
 ('F1', 'Hybrid: Q-conv + dense 80->16 + KAN [16->14] 3-bit/4b', conv_q + [head_mlp[0], kan('KAN 16->14 (3-bit in)', 16, 14, 3, 4)]),
 ('F2', 'Hybrid: KAN conv 2/4/4 + dense 80->16 + KAN [16->14] 2-bit/4b', kconv(4) + [head_mlp[0], kan('KAN 16->14 (2-bit in)', 16, 14, 2, 4)]),
]
base = total(MODELS[0][2])
if __name__ == '__main__':
    print('| id | model | params | stored bits | vs A | multiplies | lookups | additions |')
    print('|---|---|---|---|---|---|---|---|')
    for mid, name, rows in MODELS:
        t = total(rows)
        print(f"| {mid} | {name} | {t['params']:,} | {t['bits']/1000:.1f} k | {t['bits']/base['bits']*100:.0f}% | {t['mult']:,} | {t['lookups']:,} | {t['adds']:,} |")
    print()
    for mid, name, rows in MODELS[:2] + MODELS[3:4] + MODELS[8:9]:
        print(f'### {mid}: {name}')
        print('| layer | params | stored bits | multiplies | lookups | detail |'); print('|---|---|---|---|---|---|')
        for r in rows:
            print(f"| {r['layer']} | {r['params']:,} | {r['bits']:,} | {r['mult']:,} | {r['lookups']:,} | {r['desc']} |")
        print()
