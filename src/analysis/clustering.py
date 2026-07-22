"""Modularity metric for a computed-state partition on a precomputed KNN graph."""

from __future__ import annotations

import logging

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)


def compute_modularity(
    adata_processed: AnnData, labels: np.ndarray, obsp_key: str = "connectivities"
) -> float:
    """
    Modularity of the given partition on the precomputed scanpy KNN graph
    (adata.obsp[obsp_key], default the all-gene graph). Requires igraph
    (bundled with scanpy).
    """
    try:
        import igraph as ig
    except ImportError:
        logger.warning("igraph not available — skipping modularity")
        return float("nan")

    if obsp_key not in adata_processed.obsp:
        logger.warning("No precomputed neighbors graph — skipping modularity")
        return float("nan")

    A = adata_processed.obsp[obsp_key].tocoo()
    edges = list(zip(A.row.tolist(), A.col.tolist()))
    g = ig.Graph(n=adata_processed.n_obs, edges=edges, directed=False)
    g.simplify()
    return float(g.modularity(labels.tolist()))
