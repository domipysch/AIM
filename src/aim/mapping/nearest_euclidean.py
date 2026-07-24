"""Nearest spot->state mapping by Euclidean distance in lognorm shared-gene space:
assign each spot to the state centroid at the smallest Euclidean distance, producing
a one-hot P.

Unlike ``nearest`` (cosine on raw counts, magnitude-invariant), both the spot matrix
and the state profiles here are the normalize_total+log1p shared-gene values
(``OBSM_LOGNORM_SHARED_GENES`` / lognorm state centroids), so library-size and depth
differences are already normalised out and the straight Euclidean distance is a
meaningful match."""

import torch

from .base import SpotStateMapper
from .confidence import top_margin_confidence


class NearestEuclideanMapper(SpotStateMapper):
    """Nearest-centroid mapper by Euclidean distance in lognorm space (one-hot P)."""

    name = "nearest_euclidean"

    def map(self, leiden_to_state, k) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign each spot to the Euclidean-nearest state centroid in lognorm
        shared-gene space.

        Returns a one-hot P (S x K) plus a (S,) confidence: the margin of the
        winning centroid's Euclidean distance over its top runners-up.
        """
        Z = self._spatial_data_matrix_lognorm()
        M = self._state_profiles_lognorm(leiden_to_state, k)

        dist = torch.cdist(Z, M)  # (S x K) Euclidean distance
        spot_state = torch.argmin(dist, dim=1)
        P = torch.zeros(Z.shape[0], M.shape[0])
        P[torch.arange(P.shape[0]), spot_state] = 1.0
        confidence = top_margin_confidence(dist)  # Euclidean distance, lower = better
        return P, confidence
