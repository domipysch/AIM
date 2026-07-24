"""Euclidean majority-vote spot->state mapping: like ``majority_vote`` (kNN label
transfer, soft P from the nearest reference cells' states), but each spot's neighbours
are its top-N Euclidean-nearest reference cells in normalize_total+log1p shared-gene
space instead of its most cosine-similar ones on raw counts."""

import numpy as np
import torch

from adata_schema import OBSM_LOGNORM_SHARED_GENES
from analysis.utils import to_dense
from .majority_vote import MajorityVoteMapper


class MajorityVoteEuclideanMapper(MajorityVoteMapper):
    """kNN label transfer by Euclidean distance in lognorm shared-gene space."""

    name = "majority_vote_euclidean"

    def _find_neighbors(self, n: int) -> np.ndarray:
        """Indices (S, n) of each spot's top-n Euclidean-nearest reference cells in
        normalize_total+log1p shared-gene space."""
        Zs = self._spatial_data_matrix_lognorm()  # (S, G) lognorm shared
        Zc = torch.tensor(
            to_dense(self.adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES]), dtype=torch.float32
        )  # (C, G)
        dist = torch.cdist(Zs, Zc)  # (S, C) Euclidean distance
        return torch.topk(dist, n, dim=1, largest=False).indices.numpy()  # (S, n)
