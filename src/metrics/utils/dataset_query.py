from typing import List, Tuple
from pathlib import Path
import pandas as pd
from anndata import AnnData
import anndata


def get_sc_genes(sc_path: Path) -> List[str]:
    """
    Get the list of all genes in the scRNA data.

    Args:
        sc_path: Full path to sc.h5ad.

    Returns:
        List of gene ID strings in file order.
    """
    return anndata.read_h5ad(sc_path).var_names.tolist()


def get_st_genes(st_path: Path) -> List[str]:
    """
    Get the list of all genes in the ST data.

    Args:
        st_path: Full path to st.h5ad.

    Returns:
        List of gene ID strings in file order.
    """
    return anndata.read_h5ad(st_path).var_names.tolist()


def get_shared_genes(sc_path: Path, st_path: Path) -> List[str]:
    """
    Get the list of genes shared between scRNA and ST data.

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.

    Returns:
        List of gene ID strings present in both scRNA and ST data (order not guaranteed).
    """
    sc_genes = set(get_sc_genes(sc_path))
    st_genes = set(get_st_genes(st_path))
    shared_genes = list(sc_genes.intersection(st_genes))
    return shared_genes


def get_cell_annotations(sc_path: Path) -> pd.DataFrame:
    """
    Load cell annotations from sc.h5ad.

    Args:
        sc_path: Full path to sc.h5ad.

    Returns:
        DataFrame of cell annotations with cell IDs as index.
    """
    return anndata.read_h5ad(sc_path).obs


def get_z_real_and_predicted_data_only_shared_genes(
    sc_path: Path, st_path: Path, result_gep: AnnData
) -> Tuple[AnnData, AnnData]:
    """
    Load input ST data and predicted Z', filtered to shared marker genes only.

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        result_gep: Predicted gene expression as AnnData (G x S).

    Returns:
        Tuple (st_shared, result_shared): both AnnData objects of shape (S x shared_G),
        aligned to the same set of shared genes in the same order.
    """
    st_ad = anndata.read_h5ad(st_path)
    result_gep = result_gep.transpose()

    # Filter to shared marker genes, preserving the order from st_ad
    marker_genes = set(get_shared_genes(sc_path, st_path))
    common_genes = [
        g for g in st_ad.var_names if g in marker_genes and g in result_gep.var_names
    ]
    st_shared_ad = st_ad[:, common_genes].copy()
    result_shared_gep = result_gep[:, common_genes].copy()

    assert st_shared_ad.var_names.equals(
        result_shared_gep.var_names
    ), "Gene order or identity mismatch between ST data and result GEP."

    return st_shared_ad, result_shared_gep
