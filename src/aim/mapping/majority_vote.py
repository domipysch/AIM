"""Majority-vote spot->state mapping: for each ST spot take the top-N most
cosine-similar reference cells on the shared genes, tally which AIM state each of
those neighbours belongs to at the current K, and return the vote distribution."""

import logging

import numpy as np
import torch

from adata_schema import OBS_LEIDEN_ALL_GENES, UNS_SHARED_GENES
from analysis.utils import to_dense
from .base import SpotStateMapper

logger = logging.getLogger(__name__)


class MajorityVoteMapper(SpotStateMapper):
    """kNN label transfer: soft P from the top-N nearest reference cells' states."""

    eps: float = 1e-8
    name = "majority_vote"

    def __init__(self, n_neighbors: int = 10) -> None:
        self.n_neighbors = n_neighbors

    def prepare(self, adata_sc, adata_st, labels_by_k) -> None:
        """Find each spot's top-N nearest reference cells once for the whole sweep.

        The neighbour search is K-independent (only the neighbours' state labels
        change with K), so the S x n_cells cosine + top-N runs here and only the
        cached (S, N) neighbour indices and per-cell Leiden labels are reused per K.
        """
        super().prepare(adata_sc, adata_st, labels_by_k)

        shared = adata_sc.uns[UNS_SHARED_GENES]
        Zs = self._spatial_data_matrix()  # (S, G) raw shared
        Zc = torch.tensor(to_dense(adata_sc[:, shared]), dtype=torch.float32)  # (C, G)

        Zs = Zs / (Zs.norm(dim=1, keepdim=True) + self.eps)
        Zc = Zc / (Zc.norm(dim=1, keepdim=True) + self.eps)
        sim = Zs @ Zc.t()  # (S, C) cosine similarity

        n_cells = Zc.shape[0]
        n = min(self.n_neighbors, n_cells)
        if n < self.n_neighbors:
            logger.info(
                "MajorityVoteMapper: n_neighbors=%d > n_cells=%d, using %d",
                self.n_neighbors,
                n_cells,
                n,
            )
        self._n = n
        self._neighbor_idx = torch.topk(sim, n, dim=1).indices.numpy()  # (S, N)
        self._leiden = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()

    def map(self, leiden_to_state, k) -> torch.Tensor:
        """Tally the cached neighbours' states at this K into a soft P (S x K), rows summing to 1."""
        labels = np.asarray(leiden_to_state)
        neighbor_states = labels[self._leiden[self._neighbor_idx]]  # (S, N) state ids
        onehot = np.eye(k, dtype=np.float32)[neighbor_states]  # (S, N, K)
        p = onehot.sum(axis=1) / self._n  # (S, K) vote fractions
        return torch.tensor(p, dtype=torch.float32)

    def config(self) -> dict:
        return {"mapping": self.name, "n_neighbors": self.n_neighbors}
