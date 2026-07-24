"""Dispersion-scaled nearest spot->state mapping in Euclidean / lognorm space: like
``nearest_euclidean`` (Euclidean nearest centroid on the normalize_total+log1p
shared-gene values, one-hot P), but each spot->centroid Euclidean distance is divided
by the state's cell-level Euclidean dispersion (its RMS radius in lognorm space) before
the argmin, so diffuse states claim more distant spots and tight states only nearby
ones. The Euclidean analogue of ``nearest_scaled``."""

import torch

from .confidence import top_margin_confidence
from .nearest_scaled import NearestScaledMapper
from ..aggregation import assemble_state_dispersion_shared_genes_norm


class NearestEuclideanScaledMapper(NearestScaledMapper):
    """Euclidean nearest-centroid scaled by per-state lognorm-space dispersion (one-hot P)."""

    name = "nearest_euclidean_scaled"

    def map(self, leiden_to_state, k) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign each spot to the state minimizing (Euclidean distance) / dispersion
        in lognorm shared-gene space.

        Returns a one-hot P (S x K) plus a (S,) confidence: the margin of the
        winning state's scaled distance over its top runners-up.
        """
        Z = self._spatial_data_matrix_lognorm()
        M = self._state_profiles_lognorm(leiden_to_state, k)
        sigma = assemble_state_dispersion_shared_genes_norm(
            leiden_to_state, k, self.adata_sc, M, self.dispersion_shrinkage
        )
        dist = torch.cdist(Z, M)  # (S x K) Euclidean distance
        scaled = dist / (
            sigma.view(1, -1) + self.eps
        )  # broadcast dispersion over spots
        spot_state = torch.argmin(scaled, dim=1)
        P = torch.zeros(Z.shape[0], M.shape[0])
        P[torch.arange(P.shape[0]), spot_state] = 1.0
        confidence = top_margin_confidence(scaled)  # scaled distance, lower = better
        return P, confidence
