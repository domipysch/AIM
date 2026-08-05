"""Numeric metrics for analyzing a reference aligner's mapping_prob.h5ad output.

Works identically for Tangram, TACCO, and DOT — all three write the same
spots x cell-type layout with real cell-type names in var_names, so this
module needs no aligner-specific logic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from anndata import AnnData

from aim.metrics.reconstruction import predict_expression

logger = logging.getLogger(__name__)

__all__ = ["celltype_centroids", "predict_expression"]


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
