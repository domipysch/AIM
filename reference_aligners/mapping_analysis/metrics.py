"""Numeric metrics for analyzing a reference aligner's mapping_prob.h5ad output.

Works identically for Tangram, TACCO, and DOT — all three write the same
spots x cell-type layout with real cell-type names in var_names, so this
module needs no aligner-specific logic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


def celltype_centroids(
    adata: AnnData, cell_type_key: str, cell_types: list[str]
) -> pd.DataFrame:
    """
    Mean expression per cell type (rows) across all genes in `adata` (columns).

    `cell_types` fixes the row order (types with zero matching cells become
    all-zero rows) so the result aligns with a mapping's var_names.
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    labels = adata.obs[cell_type_key].astype(str).to_numpy()
    rows = []
    for ct in cell_types:
        mask = labels == ct
        if mask.sum() == 0:
            logger.warning("Cell type '%s' has no matching cells in sc data.", ct)
            rows.append(np.zeros(X.shape[1], dtype=np.float32))
        else:
            rows.append(X[mask].mean(axis=0))
    return pd.DataFrame(
        np.vstack(rows).astype(np.float32), index=cell_types, columns=adata.var_names
    )


def top_marker_genes(adata_norm: AnnData, cell_type_key: str, n_top: int) -> list[str]:
    """Top n_top marker genes per cell type (union, order preserved), via
    scanpy's rank_genes_groups on already normalized+log1p data.

    Cell types with fewer than 2 cells are skipped: the Wilcoxon test needs
    at least two samples per group, and scanpy errors out otherwise."""
    adata_copy = adata_norm.copy()

    counts = adata_copy.obs[cell_type_key].value_counts()
    valid_groups = counts[counts >= 2].index.astype(str).tolist()
    dropped = counts[counts < 2].index.astype(str).tolist()
    if dropped:
        logger.warning(
            "Skipping cell type(s) with <2 cells in marker ranking: %s",
            ", ".join(dropped),
        )

    sc.tl.rank_genes_groups(
        adata_copy, groupby=cell_type_key, groups=valid_groups, method="wilcoxon"
    )
    names = adata_copy.uns["rank_genes_groups"]["names"]
    markers: list[str] = []
    for group in names.dtype.names:
        markers.extend(list(names[group][:n_top]))
    seen: set[str] = set()
    unique_markers = []
    for g in markers:
        if g not in seen:
            seen.add(g)
            unique_markers.append(g)
    return unique_markers


def predict_expression(mapping: np.ndarray, centroids: pd.DataFrame) -> np.ndarray:
    """Predicted spot expression Z' = mapping @ centroids (S x G)."""
    return np.asarray(mapping) @ centroids.to_numpy()
