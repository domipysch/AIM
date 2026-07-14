from anndata import AnnData
import torch
import logging

logger = logging.getLogger(__name__)


def prepare_tensors_from_input(
    adata_sc: AnnData, adata_st: AnnData, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert AnnData objects to PyTorch tensors for single-cell and spatial transcriptomics data.

    Compute the following tensors, restricted to genes present in both datasets
    (the full, non-shared expression matrices are never needed by the model, so
    they are not materialized here):
    - **X_shared** (Tensor): scRNA-seq matrix restricted to genes present in both datasets.
      Shape: (C, G_shared).
    - **Z_shared** (Tensor): Spatial matrix restricted to genes present in both datasets.
      Shape: (S, G_shared).

    The 'shared' tensors are gene-aligned,
    meaning column *j* in X_shared and column *j* in Z_shared represent the same gene.

    Args:
        adata_sc (AnnData): scRNA data.
        adata_st (AnnData): ST data.
        device (torch.device): device to store the tensors.

    Returns:
        A tuple containing the gene-aligned tensors (X_shared, Z_shared).
    """
    logger.debug("Prepare tensors")

    # 1. Identify Shared Genes
    sc_genes = adata_sc.var_names
    st_genes = adata_st.var_names
    shared_genes = sc_genes.intersection(st_genes)

    # 2. Create Shared Gene Subsets
    # We use AnnData's built-in slicing to ensure gene order matches perfectly
    logger.debug("Extract shared genes")
    adata_sc_shared = adata_sc[:, shared_genes].copy()
    adata_st_shared = adata_st[:, shared_genes].copy()

    # .X might be sparse (scipy.sparse), so we use .toarray() to ensure dense format for Torch
    X_shared_raw = (
        adata_sc_shared.X.toarray()
        if hasattr(adata_sc_shared.X, "toarray")
        else adata_sc_shared.X
    )
    Z_shared_raw = (
        adata_st_shared.X.toarray()
        if hasattr(adata_st_shared.X, "toarray")
        else adata_st_shared.X
    )

    # 3. Convert to Tensors and move to Device (MPS/CPU)
    # Ensure they are float32 for Metal/MPS/CUDA compatibility
    logger.debug("Convert to tensors")
    X_shared = torch.tensor(X_shared_raw, dtype=torch.float32).to(device)
    Z_shared = torch.tensor(Z_shared_raw, dtype=torch.float32).to(device)

    return X_shared, Z_shared
