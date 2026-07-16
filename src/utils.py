from pathlib import Path
import numpy as np
import pandas as pd
import logging
import torch
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


def estimate_gpu_memory_gb(
    num_cells: int,
    num_spots: int,
    num_genes_shared: int,
    n_states: int,
) -> float:
    """
    Rough upper-bound estimate of GPU memory required (in GB).

    Only the shared-gene tensors are ever materialized on device (the full,
    non-shared expression matrices are never loaded), so the footprint is
    dominated by X_shared/Z_shared plus the small model matrices (G: L x L,
    H: S x L) and the Z' = H @ M reconstruction workspace, where L = n_states.
    """
    B32 = 4  # bytes per float32
    L = n_states

    est_bytes = B32 * (
        num_cells * num_genes_shared  # X_shared
        + num_spots * num_genes_shared  # Z_shared
        + L * num_genes_shared  # expr_sums_shared
        + 3 * num_spots * L  # H + grads + workspace
        + 3 * num_spots * num_genes_shared  # Z' reconstruction workspace
    )
    return est_bytes / (1024**3)


def arr_to_h5ad(
    arr: np.ndarray,
    path: Path,
    obs_names: list,
    var_names: list,
) -> None:
    """Save a 2D numpy array as an h5ad file with labelled obs and var axes."""
    adata = AnnData(X=arr.astype(np.float32))
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.write_h5ad(path)


def _to_numpy(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def _dense_X(adata: AnnData) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.array(X, dtype=np.float32)


def hard_assignments(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Row-wise argmax → shape (N,)."""
    return _to_numpy(matrix).argmax(axis=1)


# ---------------------------------------------------------------------------
# Shared scanpy PCA + neighbors pipeline
# ---------------------------------------------------------------------------


def run_pca_neighbors_umap(
    adata: AnnData,
    n_comps: int = 30,
    n_neighbors: int = 15,
    skip_umap: bool = False,
) -> None:
    """In-place: optional normalize/log1p → PCA → neighbors → optional UMAP."""

    # Normalize before computing PCAs
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n = min(n_comps, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    if not skip_umap:
        sc.tl.umap(adata)


def fmt_nonzero_4(x: float) -> str:
    """
    Format a numeric value for display to cap at up to four decimal places.

    Args:
        x: Input value (float)
    Returns:
        str: Formatted string
    """
    if pd.isna(x):
        return ""
    try:
        xf = float(x)
    except Exception:
        raise Exception("Input value is not convertible to float")
    if xf == 0.0:
        return "0.0"
    return f"{xf:.4f}"
