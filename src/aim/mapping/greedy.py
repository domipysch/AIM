"""
Greedy spot->state mapping: zero-parameter nearest-centroid classifier.

Each spot is assigned to the state whose (size-weighted) centroid is most
cosine-similar to it on the shared genes:

    P[s] = one_hot( argmax_k cos(Z_shared[s], M[k]) )

No training; P is one-hot by construction, so the soft and deterministic
reconstructions coincide.
"""

import torch

from .base import SpotStateMapper


class GreedyMapper(SpotStateMapper):
    """Zero-parameter nearest-centroid mapper (one-hot P)."""

    eps: float = 1e-8
    name = "greedy"

    def map(self, Z_shared: torch.Tensor, M_shared: torch.Tensor) -> torch.Tensor:
        """
        Assign nearest centroid

        Assign each spot to the state whose centroid is most cosine-similar to it,
        on the shared genes.

        Returns a one-hot spot->state assignment P (S x K).
        """

        Zn = Z_shared / (Z_shared.norm(dim=1, keepdim=True) + self.eps)
        Mn = M_shared / (M_shared.norm(dim=1, keepdim=True) + self.eps)
        sim = Zn @ Mn.t()  # (S x K) cosine similarity
        spot_state = torch.argmax(sim, dim=1)
        P = torch.zeros(Z_shared.shape[0], M_shared.shape[0])
        P[torch.arange(P.shape[0]), spot_state] = 1.0
        return P
