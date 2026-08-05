"""Load one K's saved sweep outputs for the post-mapping analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from aim.adata_schema import (
    OBSM_MAPPING_SOFT,
    OBS_MAPPING_HARD,
    OBS_MAPPING_CONFIDENCE,
    OBS_LEIDEN_ALL_GENES,
    OBS_COMPUTED_STATE,
)

logger = logging.getLogger(__name__)


def load_spot_to_state_mapping_soft_and_hard(run_dir: Path, adata_st: AnnData) -> None:
    """Load the spot->state matrix P from spot_to_state_mapping_soft.h5ad in run_dir.

    Adds: adata_st.obsm[OBSM_MAPPING_SOFT] (S x K), assigned positionally, and,
    when the mapper wrote one, adata_st.obs[OBS_MAPPING_CONFIDENCE] (S,).
    Raises ValueError if the file's spot order does not match adata_st.obs_names.
    """
    run_dir = Path(run_dir)
    mapping_path = run_dir / "spot_to_state_mapping_soft.h5ad"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Required mapping output missing: {mapping_path}")

    mapping_ad = ad.read_h5ad(mapping_path)
    if not mapping_ad.obs_names.equals(adata_st.obs_names):
        raise ValueError(
            f"Spot order in {mapping_path} does not match adata_st.obs_names."
        )
    adata_st.obsm[OBSM_MAPPING_SOFT] = np.asarray(mapping_ad.X, dtype=np.float64)
    adata_st.obs[OBS_MAPPING_HARD] = adata_st.obsm[OBSM_MAPPING_SOFT].argmax(axis=1)
    if OBS_MAPPING_CONFIDENCE in mapping_ad.obs:
        adata_st.obs[OBS_MAPPING_CONFIDENCE] = mapping_ad.obs[
            OBS_MAPPING_CONFIDENCE
        ].to_numpy()


def load_leiden_to_state(run_dir: Path) -> np.ndarray:
    """Load the subcluster->state label array (L,), values 0..K-1, from
    leiden_to_state.csv in run_dir."""
    run_dir = Path(run_dir)
    leiden_to_state_path = run_dir / "leiden_to_state.csv"
    if not leiden_to_state_path.exists():
        raise FileNotFoundError(
            f"Required mapping output missing: {leiden_to_state_path}"
        )
    return pd.read_csv(leiden_to_state_path)["state"].to_numpy()


def infer_cell_to_state_cluster(adata_sc: AnnData, leiden_to_state: np.ndarray):
    leiden_idx = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    cell_states = leiden_to_state[leiden_idx]
    adata_sc.obs[OBS_COMPUTED_STATE] = pd.Categorical(cell_states.astype(str))
