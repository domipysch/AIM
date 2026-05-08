import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import issparse  # still needed by _compute_scvi cache loading

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = ("pca", "scvi")


def compute_sc_embedding(
    adata_sc: AnnData,
    method: str,
    d: int,
    device: torch.device,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    Compute a priori cell embedding Y of shape (C, d) from scRNA-seq data.

    Args:
        adata_sc: AnnData with cells × genes expression matrix in .X.
        method: Embedding method — "pca" or "scvi".
        d: Target embedding dimension.
        device: Torch device to place the resulting tensor on.
        cache_dir: Directory containing sc.h5ad. When provided and method="scvi",
            the embedding is read from / written to sc-scvi.h5ad in that directory.

    Returns:
        Y: Float32 tensor of shape (C, d) on `device`.
    """
    if method == "pca":
        Y = _compute_pca(adata_sc, d)
    elif method == "scvi":
        Y = _compute_scvi(adata_sc, d, cache_dir=cache_dir)
    else:
        raise ValueError(
            f"Unknown sc_embedding method '{method}'. Choose from {_SUPPORTED_METHODS}."
        )

    logger.info(f"Computed sc embedding Y via {method}: shape={Y.shape}")
    return torch.tensor(Y, dtype=torch.float32, device=device)


def _compute_pca(adata_sc: AnnData, d: int) -> np.ndarray:
    import scanpy as sc

    n_cells, n_genes = adata_sc.shape
    n_components = min(d, n_cells - 1, n_genes)
    if n_components < d:
        logger.warning(
            f"Requested d={d} but only {n_components} PCA components possible "
            f"(n_cells={n_cells}, n_genes={n_genes}). Using {n_components}."
        )

    adata = adata_sc.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", inplace=True)
    sc.pp.pca(adata, n_comps=n_components, use_highly_variable=True)
    return np.asarray(adata.obsm["X_pca"], dtype=np.float32)


def _compute_scvi(
    adata_sc: AnnData, d: int, cache_dir: Optional[Path] = None
) -> np.ndarray:
    cache_path = cache_dir / "sc-scvi.h5ad" if cache_dir is not None else None

    if cache_path is not None and cache_path.exists():
        cached = ad.read_h5ad(cache_path)
        if cached.n_vars == d:
            logger.info(f"Loaded scVI embedding from cache: {cache_path}")
            X = cached.X
            if issparse(X):
                X = X.toarray()
            return np.asarray(X, dtype=np.float32)
        logger.warning(
            f"Cache at {cache_path} has d={cached.n_vars} but d={d} was requested — recomputing."
        )

    try:
        import scvi
    except ImportError as e:
        raise ImportError(
            "scVI is required for sc_embedding.method='scvi'. "
            "Install it with: pip install scvi-tools"
        ) from e

    scvi.model.SCVI.setup_anndata(adata_sc)
    model = scvi.model.SCVI(adata_sc, n_latent=d)
    model.train(max_epochs=200)
    embedding = model.get_latent_representation().astype(np.float32)

    if cache_path is not None:
        cache_adata = AnnData(X=embedding)
        cache_adata.obs_names = adata_sc.obs_names
        cache_adata.write_h5ad(cache_path)
        logger.info(f"Saved scVI embedding to cache: {cache_path}")

    return embedding
