"""Biological / topological metrics for the post-mapping analysis.

Three questions, each producing a scalar objective (every permutation-tested
metric is reported with an observed value AND a null z-score; the z-score is the
dataset-comparable objective, since raw purity / Moran's I / cosine values are
not comparable across pairs with different numbers of states):

1. Spatial organisation of the mapped spots (P's argmax state per spot)
   - Local Spatial Purity (LSP): mean fraction of the k nearest spatial
     neighbours sharing the same mapped state — local compactness.
   - Moran's I averaged over the per-state binary indicators — global spatial
     clustering (via squidpy).
   - Null: shuffle the spot-state labels N_PERM times.

2. Coherence of the subclusters merged into one computed state
   - For each computed state that aggregates >=2 subclusters: are those
     subclusters' shared-gene centroids MORE mutually similar (higher mean
     pairwise cosine) than a same-sized random draw of subclusters from OTHER
     states? If yes (high z-score, low p), the merge looks coherent in the gene
     subspace the method actually operates in.
   - Null: N_PERM random same-sized draws of subclusters from other states.

3. Modularity of the computed-state partition on a precomputed KNN graph.

These are generic given their inputs (label arrays, centroids, a graph), so they
live in ``metrics`` rather than in the AIM-specific ``analysis`` package.
"""

from __future__ import annotations

import logging
from typing import Callable

import anndata as ad
import numpy as np
import squidpy as sq
from anndata import AnnData

logger = logging.getLogger(__name__)


def leiden_state_groups(labels_k: np.ndarray) -> dict[int, list[int]]:
    """Map each computed state to the list of subclusters merged into it.

    labels_k[l] is the state subcluster l was merged into by the tree cut
    (values 0..k-1) — the exact grouping the model learned, no recomputation.
    """
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
    """
    Modularity of the given partition on the precomputed scanpy KNN graph
    (adata.obsp[obsp_key], default the all-gene graph). Requires igraph
    (bundled with scanpy). Returns NaN if igraph or the graph is unavailable.
    """
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
