"""Coherence and modularity metrics over computed states: grouping subclusters
by state, pairwise cosine statistics of their centroids, and graph modularity of
a partition."""

from __future__ import annotations

import logging
from typing import Callable

import anndata as ad
import numpy as np
import squidpy as sq
from anndata import AnnData

logger = logging.getLogger(__name__)


def leiden_state_groups(labels_k: np.ndarray) -> dict[int, list[int]]:
    """Map each computed state to the list of subclusters assigned to it, given
    ``labels_k[l]`` = state of subcluster ``l`` (values 0..k-1)."""
    groups: dict[int, list[int]] = {}
    for leiden_cluster, state in enumerate(np.asarray(labels_k)):
        groups.setdefault(int(state), []).append(int(leiden_cluster))
    return groups


def _pairwise_cosine_stats(vecs: np.ndarray) -> tuple[float, float]:
    """(mean, median) pairwise cosine similarity over all unique row pairs of ``vecs``."""
    n = len(vecs)
    if n < 2:
        return 1.0, 1.0
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / (norm + 1e-12)
    sim = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    sims = sim[iu]
    return float(sims.mean()), float(np.median(sims))


def compute_modularity(
    adata: AnnData, labels: np.ndarray, obsp_key: str = "connectivities"
) -> float:
    """Modularity of ``labels`` on the precomputed KNN graph in
    ``adata.obsp[obsp_key]``. Requires igraph; returns NaN if igraph or the graph
    is unavailable."""
    try:
        import igraph as ig
    except ImportError:
        logger.warning("igraph not available — skipping modularity")
        return float("nan")

    if obsp_key not in adata.obsp:
        logger.warning("No precomputed neighbors graph — skipping modularity")
        return float("nan")

    A = adata.obsp[obsp_key].tocoo()
    edges = list(zip(A.row.tolist(), A.col.tolist()))
    g = ig.Graph(n=adata.n_obs, edges=edges, directed=False)
    g.simplify()
    return float(g.modularity(labels.tolist()))
