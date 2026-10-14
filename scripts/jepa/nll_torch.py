"""PyTorch port of conditional_nll.custom_loss_v2 (softplus diagonal, unclipped, batch SUM).

Same 14-output layout as the TF loss: mu = p[:, 0:8:2], diag = minval + softplus(p[:, 1:8:2]),
off-diagonals p[:, 8:14] filling L row-wise exactly as conditional_nll.scale_tril does.
Verified against the TF implementation by check_nll_port.py.
"""
import math, torch, torch.nn.functional as F
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)

def nll_terms(y, p, minval=1e-9):
    # Explicit forward substitution for the 4x4 lower-triangular L (rows filled as in
    # conditional_nll.scale_tril). Same arithmetic as a triangular solve, but plain
    # elementwise ops: torch.linalg.solve_triangular's CUDA backward JIT-compiles a
    # Triton kernel that needs CUDA headers this pod does not have.
    mu = p[:, 0:8:2]; d = minval + F.softplus(p[:, 1:8:2]); c = p[:, 8:14]
    r = y - mu
    z0 = r[:, 0] / d[:, 0]
    z1 = (r[:, 1] - c[:, 0] * z0) / d[:, 1]
    z2 = (r[:, 2] - c[:, 1] * z0 - c[:, 2] * z1) / d[:, 2]
    z3 = (r[:, 3] - c[:, 3] * z0 - c[:, 4] * z1 - c[:, 5] * z2) / d[:, 3]
    z = torch.stack([z0, z1, z2, z3], 1)
    return 0.5 * z ** 2 + torch.log(d) + HALF_LOG_2PI           # (B, 4), rows sum to the joint

def custom_loss_v2(y, p):
    return nll_terms(y, p).sum()
