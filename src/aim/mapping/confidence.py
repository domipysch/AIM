"""Per-spot mapping confidence in [0, 1]: how decisively each spot was assigned.

Confidence is optional — a mapper returns ``None`` alongside its P when it does
not define one (the external ``reference`` mappers). Where a mapper does define
it, higher means a more decisive assignment.
"""

import math

import torch

# How many top candidate states the margin-based confidence looks at.
N_TOP_STATES = 4


def top_margin_confidence(score: torch.Tensor, n: int = N_TOP_STATES) -> torch.Tensor:
    """Relative margin of the best (lowest) score over its closest runners-up.

    ``score`` is (S x K) with lower = better (a distance). For each spot take the
    ``n`` smallest scores d1 <= d2 <= ... <= dn and return
    ``1 - d1 / mean(d2..dn)`` in [0, 1]: ~1 when the winner is far ahead of the
    field, ~0 when the top candidates are (near-)tied. With a single state (K=1)
    there is no competition, so confidence is 1.
    """
    eps = 1e-8
    n_spots, k = score.shape
    if k == 1:
        return torch.ones(n_spots, dtype=torch.float32)
    n = min(n, k)
    top = torch.topk(score, n, dim=1, largest=False).values  # (S x n), ascending
    best = top[:, 0]
    rest = top[:, 1:].mean(dim=1)
    conf = 1.0 - best / (rest + eps)
    return conf.clamp(0.0, 1.0).to(torch.float32)


def entropy_confidence(p: torch.Tensor) -> torch.Tensor:
    """One-hotness of each soft row via normalized Shannon entropy.

    ``p`` is (S x K) with rows summing to 1. Returns ``1 - H(row) / log(K)`` in
    [0, 1]: 1 when the row is one-hot (all mass on one state), 0 when the vote is
    uniform across states. With a single state (K=1) the row is trivially
    one-hot, so confidence is 1.
    """
    eps = 1e-12
    n_spots, k = p.shape
    if k == 1:
        return torch.ones(n_spots, dtype=torch.float32)
    entropy = -(p * torch.log(p + eps)).sum(dim=1)
    conf = 1.0 - entropy / math.log(k)
    return conf.clamp(0.0, 1.0).to(torch.float32)
