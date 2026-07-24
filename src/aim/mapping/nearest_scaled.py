"""Dispersion-scaled nearest spot->state mapping: like ``nearest`` (nearest cosine
centroid, one-hot P), but each spot->centroid cosine distance is divided by the
state's cell-level cosine dispersion before the argmin, so diffuse states claim
more distant spots and tight states only claim nearby ones."""

import torch

from .base import SpotStateMapper
from .confidence import top_margin_confidence
from ..aggregation import assemble_state_dispersion_shared_genes


class NearestScaledMapper(SpotStateMapper):
    """Nearest-centroid mapper scaled by per-state cosine dispersion (one-hot P)."""

    eps: float = 1e-8
    name = "nearest_scaled"

    def __init__(self, dispersion_shrinkage: float = 1.0) -> None:
        self.dispersion_shrinkage = dispersion_shrinkage

    def map(self, leiden_to_state, k) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign each spot to the state minimizing (1 - cos) / dispersion.

        Returns a one-hot P (S x K) plus a (S,) confidence: the margin of the
        winning state's scaled distance over its top runners-up (analogous to
        ``nearest``, but on the dispersion-scaled distance that decides the cut).
        """

        Z_shared = self._spatial_data_matrix()
        M_shared = self._state_profiles(leiden_to_state, k)
        sigma = assemble_state_dispersion_shared_genes(
            leiden_to_state, k, self.adata_sc, M_shared, self.dispersion_shrinkage
        )
        Zn = Z_shared / (Z_shared.norm(dim=1, keepdim=True) + self.eps)
        Mn = M_shared / (M_shared.norm(dim=1, keepdim=True) + self.eps)
        dist = 1.0 - Zn @ Mn.t()  # (S x K) cosine distance
        scaled = dist / (
            sigma.view(1, -1) + self.eps
        )  # broadcast dispersion over spots
        spot_state = torch.argmin(scaled, dim=1)
        P = torch.zeros(Z_shared.shape[0], M_shared.shape[0])
        P[torch.arange(P.shape[0]), spot_state] = 1.0
        confidence = top_margin_confidence(scaled)  # scaled distance, lower = better
        return P, confidence

    def config(self) -> dict:
        return {"mapping": self.name, "dispersion_shrinkage": self.dispersion_shrinkage}
