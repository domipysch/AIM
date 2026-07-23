"""
Load one K's saved sweep outputs back for the post-mapping analysis.

    load_mapping        run_dir -> P onto adata_st.obsm[OBSM_MAPPING_SOFT]
    load_leiden_to_state  run_dir -> subcluster->state label array (L,)

These read the exact per-K disk layout ``aim.io.write_run_outputs`` writes, so
the analysis consumes a sweep folder unchanged. The metric computations these
feed (centroid assembly, reconstruction, one-hotness, biology) live in the
``metrics`` package; this module is pure I/O.
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
        run_dir: Folder containing spot_to_state_mapping.h5ad (one K_<kkk> folder).
        adata_st: the ST AnnData the mapping was computed against.
                  spot_to_state_mapping.h5ad's obs order is written directly from
                  this object's obs_names (see aim.io.write_run_outputs), so P is
                  assigned into obsm positionally — checked against
                  adata_st.obs_names to catch any reordering since.

    Raises:
        ValueError: if the spot order does not match adata_st.obs_names.
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
    Load one K's subcluster -> state label array from leiden_to_state.csv.

    Returns:
        labels_k: subcluster -> state label array (L,), values 0..K-1.
    """
    run_dir = Path(run_dir)
    leiden_to_state_path = run_dir / "leiden_to_state.csv"
    if not leiden_to_state_path.exists():
        raise FileNotFoundError(
            f"Required mapping output missing: {leiden_to_state_path}"
        )
    return pd.read_csv(leiden_to_state_path)["state"].to_numpy()
