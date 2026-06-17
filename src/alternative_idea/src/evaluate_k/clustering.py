"""Modularity metric and Leiden reference clustering."""

from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
from anndata import AnnData

from ..utils import run_pca_neighbors_umap

logger = logging.getLogger(__name__)


def compute_modularity(adata_processed: AnnData, labels: np.ndarray) -> float:
    """
    Modularity of the given partition on the precomputed scanpy KNN graph
    (adata.obsp['connectivities']).  Requires igraph (bundled with scanpy).
    """
    try:
        import igraph as ig
    except ImportError:
        logger.warning("igraph not available — skipping modularity")
        return float("nan")

    if "connectivities" not in adata_processed.obsp:
        logger.warning("No precomputed neighbors graph — skipping modularity")
        return float("nan")

    A = adata_processed.obsp["connectivities"].tocoo()
    edges = list(zip(A.row.tolist(), A.col.tolist()))
    g = ig.Graph(n=adata_processed.n_obs, edges=edges, directed=False)
    g.simplify()
    return float(g.modularity(labels.tolist()))


def run_leiden_shared_genes(
    adata_processed: AnnData,
    shared_genes: list[str],
    resolution: float,
) -> tuple[np.ndarray, AnnData]:
    """
    Leiden clustering using only the sc/st shared genes.

    Subsets adata_processed to shared_genes, recomputes PCA/neighbors/UMAP
    on that subset, then runs Leiden.

    Returns
    -------
    labels         : Integer cluster labels, shape (n_cells,).
    adata_shared   : AnnData subset (shared genes only) with UMAP + Leiden.
    """
    available = [g for g in shared_genes if g in adata_processed.var_names]
    if len(available) < 2:
        raise ValueError(
            f"Too few shared genes found in adata ({len(available)}); cannot cluster."
        )
    logger.info("Leiden on shared genes: %d genes", len(available))

    adata_shared = adata_processed[:, available].copy()
    run_pca_neighbors_umap(adata_shared)
    sc.tl.leiden(adata_shared, resolution=resolution, key_added="_leiden_shared")
    labels = adata_shared.obs["_leiden_shared"].astype(int).values
    logger.info(
        "Leiden on shared genes (resolution=%.2f): %d clusters",
        resolution,
        len(np.unique(labels)),
    )
    return labels, adata_shared


def run_leiden_clustering(
    adata_sc: AnnData,
    resolution: float,
) -> tuple[np.ndarray, AnnData]:
    """
    Standalone Leiden clustering on the sc data.

    Returns
    -------
    labels          : Integer cluster labels, shape (n_cells,).
    adata_processed : Working copy of adata_sc with UMAP + Leiden stored.
    """
    adata = adata_sc.copy()
    run_pca_neighbors_umap(adata, skip_umap=True)
    sc.tl.leiden(adata, resolution=resolution, key_added="_leiden_ref")
    labels = adata.obs["_leiden_ref"].astype(int).values
    logger.info(
        "Leiden clustering (resolution=%.2f): %d clusters",
        resolution,
        len(np.unique(labels)),
    )
    return labels, adata
