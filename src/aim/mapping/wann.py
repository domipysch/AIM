"""Weighted Adaptive Nearest Neighbor (WANN) spot->state mapping.

Ports the reliability-weighted adaptive kNN of Di Salvo et al. (TMLR 2025,
*An Embedding is Worth a Thousand Noisy Labels*) to the AIM K-sweep. Reference
scRNA cells play the role of the (potentially noisy) training set and ST spots
the test set; a cell's label is its AIM state at the current K.

Two pieces per K:

1. **Label reliability** eta_c for every reference cell (Eq. 1 / Algorithm 1):
   the inverse of the smallest neighbourhood size k' (searched over the odd grid
   k' in {K_MIN, K_MIN+2, ..., K_MAX}) at which a k'-NN vote of the *other*
   reference cells recovers the cell's own state. Deep-in-class cells resolve at a
   small k' (high reliability); boundary/ambiguous cells need a large k' (low
   reliability); if none in range works, eta_c = 1/k_max.
2. **WANN** per spot (Eq. 2-4): the spot inherits the neighbourhood size
   k_T = 1/eta_n of its single nearest reference cell n, then votes over its
   top-k_T reference neighbours weighted by each neighbour's reliability eta_i.

The neighbour *rankings* (ref<->ref and spot<->ref, cosine on the shared genes)
depend only on the embeddings, not on K -- so they are computed once in
``prepare`` and only the per-cell state labels change per K. This replaces the
fixed-N ``majority_vote`` mapper and takes no ``n_neighbors`` parameter.
"""

import logging

import numpy as np

from aim.adata_schema import (
    OBS_LEIDEN_ALL_GENES,
    UNS_SHARED_GENES,
)
from aim.analysis.utils import to_dense
from .base import SpotStateMapper
from .confidence import entropy_confidence

logger = logging.getLogger(__name__)

# Reliability search grid (paper defaults): smallest / largest neighbourhood size
# probed for correct self-classification, stepped by 2 to keep odd sizes.
K_MIN = 11
K_MAX = 51
K_STEP = 2

# Rank tie-break weight: a neighbour at rank j (0 = nearest) contributes
# 1 + RANK_EPS * (nn_max - j) to a class tally, so on an otherwise integer vote
# tie the class with the *nearer* members wins. Kept small enough that the summed
# bonus over any window stays < 1 and never overturns a real majority.
RANK_EPS = 1e-4


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest values per row of ``scores`` (rows x N), each
    row ordered largest-first. numpy has no ``torch.topk``: ``argpartition`` picks
    the top ``k`` (unordered), then a stable ``argsort`` orders those ``k`` by
    value descending. Returns a (rows x k) index array."""
    k = min(k, scores.shape[1])
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


class WANNMapper(SpotStateMapper):
    """Reliability-weighted adaptive-kNN label transfer (soft P over states)."""

    eps: float = 1e-8
    name = "wann"

    def prepare(self, adata_sc, adata_st, labels_by_k) -> None:
        """Cache the K-independent cosine neighbour structure for the whole sweep.

        Computes, once: each reference cell's top-``nn_max`` nearest reference
        cells (self excluded) for the reliability search, and each spot's
        top-``nn_max`` nearest reference cells for the WANN vote (the spot's
        single nearest cell is column 0). Only the per-cell state labels change
        per K, so ``map`` reuses these caches.
        """
        super().prepare(adata_sc, adata_st, labels_by_k)

        n_cells = adata_sc.n_obs
        # Clamp the grid to what the reference can support (self excluded, so at
        # most n_cells - 1 neighbours). ``nn_max`` is the largest k' we ever probe.
        nn_max = min(K_MAX, max(1, n_cells - 1))
        k_lo = min(K_MIN, nn_max)
        grid = list(range(k_lo, nn_max + 1, K_STEP)) or [nn_max]
        nn_max = grid[-1]
        if grid[-1] != K_MAX:
            logger.info(
                "%s: reliability grid clamped to k'=%d..%d (n_cells=%d)",
                type(self).__name__,
                grid[0],
                grid[-1],
                n_cells,
            )
        self._grid = grid
        self._nn_max = nn_max

        shared = self.adata_sc.uns[UNS_SHARED_GENES]
        Zs = self._spatial_data_matrix()  # (S, G) raw shared ST
        Zc = np.asarray(
            to_dense(self.adata_sc[:, shared]), dtype=np.float32
        )  # (C, G) raw shared reference
        Zs = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + self.eps)
        Zc = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + self.eps)

        # (C, nn_max) ref->ref and (S, nn_max) spot->ref nearest-neighbour indices.
        self._ref_nbr_idx = self._cosine_topk(Zc, Zc, nn_max, exclude_self=True)
        self._spot_nbr_idx = self._cosine_topk(Zs, Zc, nn_max, exclude_self=False)
        self._spot_nn = self._spot_nbr_idx[:, 0]  # (S,) each spot's nearest cell

        self._leiden = self.adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()

    @staticmethod
    def _cosine_topk(
        Q: np.ndarray,
        R: np.ndarray,
        k: int,
        exclude_self: bool,
        batch_rows: int = 4096,
    ) -> np.ndarray:
        """Top-``k`` most cosine-similar rows of ``R`` for each row of ``Q``.

        ``Q`` and ``R`` must be L2-normalised so ``Q @ R.T`` is cosine similarity.
        Batched over ``Q`` rows to avoid materialising the full (|Q| x |R|)
        similarity matrix (|R| = n_cells can be large). When ``exclude_self``,
        ``Q is R`` and the diagonal (a cell with itself) is masked out. Returns a
        (|Q|, k) int index array of ``R`` indices, nearest first.
        """
        out = []
        for start in range(0, Q.shape[0], batch_rows):
            q = Q[start : start + batch_rows]
            sim = q @ R.T  # (b, |R|)
            if exclude_self:
                rows = np.arange(q.shape[0])
                cols = np.arange(start, start + q.shape[0])
                sim[rows, cols] = -np.inf
            out.append(_topk_indices(sim, k))
        return np.concatenate(out, axis=0)

    def _reliability(self, y: np.ndarray, k: int) -> np.ndarray:
        """Per-cell reliability eta (C,) at this K via Algorithm 1.

        ``y`` is the (C,) state id of every reference cell at this K. For each cell
        walks the odd grid k'=grid[0], grid[1], ...; eta = 1/k' at the first k'
        whose reliability-agnostic plurality over the cell's k' nearest neighbours
        equals the cell's own state, else 1/grid[-1]. Plurality ties break toward
        the class with the nearer members (see ``RANK_EPS``).
        """
        n_cells = y.shape[0]
        nbr_states = y[self._ref_nbr_idx]  # (C, nn_max) neighbour state ids
        counts = np.zeros((n_cells, k), dtype=np.float32)
        eta = np.full(n_cells, 1.0 / self._grid[-1], dtype=np.float32)
        resolved = np.zeros(n_cells, dtype=bool)
        rows = np.arange(n_cells)

        prev = 0
        for kp in self._grid:
            # Extend the running tally to the first kp neighbours (grid steps by 2,
            # so this adds K_STEP columns per iteration, nn_max columns in total).
            for j in range(prev, kp):
                w = 1.0 + RANK_EPS * (self._nn_max - j)
                np.add.at(counts, (rows, nbr_states[:, j]), np.float32(w))
            prev = kp
            pred = np.argmax(counts, axis=1)
            newly = (pred == y) & ~resolved
            eta[newly] = np.float32(1.0 / kp)
            resolved |= newly
            if resolved.all():
                break
        return eta

    def map(self, leiden_to_state, k) -> tuple[np.ndarray, np.ndarray]:
        """WANN spot->state soft assignment at this K.

        Labels every reference cell by its state, scores each cell's reliability,
        then for each spot inherits k_T = round(1/eta) of its nearest cell and
        casts a reliability-weighted vote over its top-k_T reference neighbours.
        Returns P (S x K, rows summing to 1) and the ``entropy_confidence`` of each
        vote row (1 = unanimous, 0 = uniform across states).
        """
        labels = np.asarray(leiden_to_state, dtype=np.int64)
        y = labels[self._leiden]  # (C,) state id per reference cell

        eta = self._reliability(y, k)  # (C,)

        # Adaptive neighbourhood size per spot: the k' of its nearest cell
        # (1/eta), clamped to what was cached. eta = 1/k' exactly, so round() is a
        # safe float->int recovery.
        k_t = np.clip(
            np.round(1.0 / eta[self._spot_nn]).astype(np.int64), 1, self._nn_max
        )

        n_spots = self._spot_nbr_idx.shape[0]
        nbr_states = y[self._spot_nbr_idx]  # (S, nn_max)
        nbr_eta = eta[self._spot_nbr_idx]  # (S, nn_max)
        # Keep only each spot's first k_t neighbours, weight them by reliability.
        keep = np.arange(self._nn_max)[None, :] < k_t[:, None]  # (S, nn_max)
        weights = (nbr_eta * keep).astype(np.float32)  # (S, nn_max)

        P = np.zeros((n_spots, k), dtype=np.float32)
        rows = np.broadcast_to(np.arange(n_spots)[:, None], nbr_states.shape)
        np.add.at(P, (rows, nbr_states), weights)
        P = P / np.clip(P.sum(axis=1, keepdims=True), self.eps, None)
        return P, entropy_confidence(P)
