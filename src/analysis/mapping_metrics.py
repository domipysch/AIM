"""
Mapping-level metrics for AIM's decoupled post-mapping analysis: loading the
raw P (spot_to_state_mapping.h5ad) matrix and the Leiden subcluster -> state
label array (leiden_to_state.csv), hardening P (argmax), validating the hard
mapping against the (always-hard, by construction) tree cut, and assembling
per-state gene expression centroids for the cosine-similarity reconstruction.

This is AIM-specific (unlike metrics.onehot / metrics.cossim, which are
generic and shared with reference_aligners/mapping_analysis) because AIM has
a two-level structure — Leiden subclusters merged into computed states via
the tree cut, spots mapped onto those states via P — that reference aligners
don't have.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from adata_schema import OBSM_MAPPING_SOFT

logger = logging.getLogger(__name__)


def load_mapping(run_dir: Path, adata_st: AnnData) -> None:
    """
    Load one K's raw P (spot -> state) matrix directly onto
    ``adata_st.obsm[OBSM_MAPPING_SOFT]`` (S x K).

    Args:
        run_dir: Folder containing spot_to_state_mapping.h5ad (as written by
                  main.py), i.e. one K_<kkk> sweep folder.
        adata_st: the same ST AnnData the mapping was computed against.
                  spot_to_state_mapping.h5ad's obs order is written directly
                  from this object's obs_names (see aim.io.write_run_outputs),
                  so P is assigned into obsm positionally — checked against
                  adata_st.obs_names to catch any reordering since.

    Raises:
        ValueError: if spot_to_state_mapping.h5ad's spot order doesn't match
                    adata_st.obs_names.
    """
    run_dir = Path(run_dir)
    mapping_path = run_dir / "spot_to_state_mapping.h5ad"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Required mapping output missing: {mapping_path}")

    mapping_ad = ad.read_h5ad(mapping_path)
    if not mapping_ad.obs_names.equals(adata_st.obs_names):
        raise ValueError(
            f"Spot order in {mapping_path} does not match adata_st.obs_names."
        )
    adata_st.obsm[OBSM_MAPPING_SOFT] = np.asarray(mapping_ad.X, dtype=np.float64)


def load_leiden_to_state(run_dir: Path) -> np.ndarray:
    """
    Load one K's Leiden subcluster -> state label array from leiden_to_state.csv.

    Args:
        run_dir: Folder containing leiden_to_state.csv (as written by
                  main.py), i.e. one K_<kkk> sweep folder.

    Returns:
        labels_k: Leiden subcluster -> state label array (L,), values 0..K-1.
    """
    run_dir = Path(run_dir)
    leiden_to_state_path = run_dir / "leiden_to_state.csv"
    if not leiden_to_state_path.exists():
        raise FileNotFoundError(
            f"Required mapping output missing: {leiden_to_state_path}"
        )
    return pd.read_csv(leiden_to_state_path)["state"].to_numpy()


def assemble_state_centroids(
    labels_k: np.ndarray,
    k: int,
    expr_sums: np.ndarray,
    sizes: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Assemble per-state gene expression centroids from the subcluster->state
    label array and the fixed Leiden-cluster expression sums/sizes (mirrors
    main.py's old assemble_state_gep, and is the numpy mirror of
    ``aim.aggregation.assemble_state_profiles_shared_genes``, which does the
    same on torch for the in-sweep computation):

        M[s] = (sum_{l: labels_k[l]=s} expr_sums[l]) / (sum_{l: labels_k[l]=s} sizes[l])

    Args:
        labels_k: Leiden subcluster -> state label array (L,), values 0..k-1.
        k: number of computed states (rows of the returned M).
        expr_sums: summed expression per Leiden cluster (L x G_genes).
        sizes: number of cells per Leiden cluster (L,).
        eps: added to the denominator to avoid division by zero for states
             with no Leiden-cluster support.

    Returns:
        M: state gene expression profiles (K x G_genes).
    """
    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=expr_sums.dtype)
    np.add.at(state_sums, labels_k, expr_sums)
    state_sizes = np.zeros(k, dtype=sizes.dtype)
    np.add.at(state_sizes, labels_k, sizes)
    return state_sums / (state_sizes[:, None] + eps)


def predict_expression(mapping: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Predicted spot expression Z' = mapping @ centroids (S x G_genes)."""
    return np.asarray(mapping) @ np.asarray(centroids)


def save_matrix_h5ad(
    matrix: np.ndarray, obs_names: list[str], var_names: list[str], path: Path
) -> None:
    """Save a plain matrix as an h5ad with the given obs/var labels."""
    AnnData(
        X=np.asarray(matrix, dtype=np.float32),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    ).write_h5ad(path)
