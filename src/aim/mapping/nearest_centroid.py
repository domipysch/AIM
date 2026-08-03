"""Nearest spot->state mapping: assign each spot to its most cosine-similar state
centroid, producing a one-hot P."""

import torch

from .base import SpotStateMapper
from .confidence import top_margin_confidence


class NearestCentroidMapper(SpotStateMapper):
    """Zero-parameter nearest-centroid mapper (one-hot P)."""

    eps: float = 1e-8
    name = "nearest_centroid"

    def map(self, leiden_to_state, k) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign each spot to its most cosine-similar state centroid.

        Returns a one-hot P (S x K) plus a (S,) confidence: the margin of the
        winning centroid's cosine distance over its top runners-up.
        """

        Z_shared = self._spatial_data_matrix()
        M_shared = self._state_profiles(leiden_to_state, k)

        Zn = Z_shared / (Z_shared.norm(dim=1, keepdim=True) + self.eps)
        Mn = M_shared / (M_shared.norm(dim=1, keepdim=True) + self.eps)
        sim = Zn @ Mn.t()  # (S x K) cosine similarity
        spot_state = torch.argmax(sim, dim=1)
        P = torch.zeros(Z_shared.shape[0], M_shared.shape[0])
        P[torch.arange(P.shape[0]), spot_state] = 1.0
        confidence = top_margin_confidence(1.0 - sim)  # cosine distance, lower = better
        return P, confidence
