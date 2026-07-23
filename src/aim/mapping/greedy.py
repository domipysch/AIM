"""Greedy spot->state mapping: assign each spot to its most cosine-similar state
centroid, producing a one-hot P."""

import torch

from .base import SpotStateMapper


class GreedyMapper(SpotStateMapper):
    """Zero-parameter nearest-centroid mapper (one-hot P)."""

    eps: float = 1e-8
    name = "greedy"

    def map(self, Z_shared: torch.Tensor, M_shared: torch.Tensor) -> torch.Tensor:
        """Assign each spot to its most cosine-similar state centroid; returns a one-hot P (S x K)."""

        Zn = Z_shared / (Z_shared.norm(dim=1, keepdim=True) + self.eps)
        Mn = M_shared / (M_shared.norm(dim=1, keepdim=True) + self.eps)
        sim = Zn @ Mn.t()  # (S x K) cosine similarity
        spot_state = torch.argmax(sim, dim=1)
        P = torch.zeros(Z_shared.shape[0], M_shared.shape[0])
        P[torch.arange(P.shape[0]), spot_state] = 1.0
        return P
