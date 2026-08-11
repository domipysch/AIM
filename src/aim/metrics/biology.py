"""Coherence and modularity metrics over computed states: grouping start clusters
by state, pairwise cosine statistics of their centroids, and graph modularity of
a partition."""

from __future__ import annotations

import logging

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)


def start_cluster_state_groups(labels_k: np.ndarray) -> dict[int, list[int]]:
    """Map each computed state to the list of start clusters assigned to it, given
    ``labels_k[l]`` = state of start cluster ``l`` (values 0..k-1)."""
    groups: dict[int, list[int]] = {}
    for start_cluster, state in enumerate(np.asarray(labels_k)):
        groups.setdefault(int(state), []).append(int(start_cluster))
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


def _igraph_from_obsp(adata: AnnData, obsp_key: str):
    """The undirected simple igraph of ``adata.obsp[obsp_key]``, or ``None``.

    ``None`` (with a warning) when igraph is not installed or the graph has not
    been computed, so every caller can degrade to NaN.
    """
    try:
        import igraph as ig
    except ImportError:
        logger.warning("igraph not available — skipping modularity")
        return None

    if obsp_key not in adata.obsp:
        logger.warning("No precomputed neighbors graph — skipping modularity")
        return None

    A = adata.obsp[obsp_key].tocoo()
    edges = list(zip(A.row.tolist(), A.col.tolist()))
    g = ig.Graph(n=adata.n_obs, edges=edges, directed=False)
    g.simplify()
    return g


def compute_modularity(
    adata: AnnData, labels: np.ndarray, obsp_key: str = "connectivities"
) -> float:
    """Modularity of ``labels`` on the precomputed KNN graph in
    ``adata.obsp[obsp_key]``. Requires igraph; returns NaN if igraph or the graph
    is unavailable."""
    g = _igraph_from_obsp(adata, obsp_key)
    if g is None:
        return float("nan")
    return float(g.modularity(np.asarray(labels).tolist()))
