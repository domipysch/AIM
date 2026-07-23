"""Spatial organisation of the mapped spots for the post-mapping analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import squidpy as sq

from adata_schema import (
    OBS_MAPPING_HARD,
    OBS_MAPPING_STATE_CAT,
    OBSM_SPATIAL,
    OBSP_SPATIAL_CONNECTIVITIES,
    UNS_NHOOD_ENRICHMENT,
)
from metrics.topology import local_spatial_purity, permutation_test

logger = logging.getLogger(__name__)


K_SPATIAL = 6  # neighbours for the squidpy spatial KNN graph
N_PERM_SPATIAL = 200  # permutations for the spatial-organisation null


def spatial_connectivities(adata_st: ad.AnnData, k: int):
    """Squidpy spatial KNN connectivity matrix, built once and cached on adata_st.

    Depends only on the spot coordinates, which are constant across the K-sweep,
    so it is computed on the first K and reused for every later K. Serves both the
    local spatial purity metric and the neighbourhood enrichment. Returns the
    (n_spots x n_spots) sparse adjacency in adata_st.obsp[OBSP_SPATIAL_CONNECTIVITIES].
    """
    if OBSP_SPATIAL_CONNECTIVITIES not in adata_st.obsp:
        sq.gr.spatial_neighbors(adata_st, coord_type="generic", n_neighs=k)
    return adata_st.obsp[OBSP_SPATIAL_CONNECTIVITIES]


def analyse_spatial_organization(
    adata_st: ad.AnnData,
    output_data_dir: Path,
    k: int = K_SPATIAL,
    n_perm: int = N_PERM_SPATIAL,
    seed: int = 0,
):
    """Local spatial purity + neighbourhood enrichment of the mapped spot states.

    Local spatial purity comes with a permutation-null z-score; neighbourhood
    enrichment (squidpy) summarises how much each state preferentially borders
    itself. Individually undefined metrics come back as NaN/None; asserts
    len(coords) == len(spot_states) and len(spot_states) > k.

    Requires: adata_st.obs[OBS_MAPPING_HARD], adata_st.obsm[OBSM_SPATIAL].
    Writes topology_metrics.json under output_data_dir.
    """
    spot_states = adata_st.obs[OBS_MAPPING_HARD].to_numpy()
    coords = np.asarray(adata_st.obsm[OBSM_SPATIAL])

    result: dict = {
        "n_spots": int(len(spot_states)),
        "n_mapped_states": int(len(np.unique(spot_states))),
        "k": int(k),
        "n_perm": int(n_perm),
        "local_purity": None,
        "nhood_enrichment": None,
    }
    assert len(coords) == len(
        spot_states
    ), "Spatial organisation skipped: coordinates length mismatch "
    assert len(spot_states) > k, "too few spots"

    rng = np.random.default_rng(seed)

    conn = spatial_connectivities(adata_st, k)
    obs_lsp = local_spatial_purity(spot_states, conn)
    result["local_purity"] = permutation_test(
        obs_lsp, spot_states, lambda l: local_spatial_purity(l, conn), n_perm, rng
    )

    result["nhood_enrichment"] = _nhood_enrichment(
        adata_st, spot_states, k, n_perm, seed
    )

    with open(output_data_dir / "topology_metrics.json", "w") as f:
        json.dump(result, f, indent=4)


def _nhood_enrichment(
    adata_st: ad.AnnData,
    spot_states: np.ndarray,
    k: int,
    n_perm: int,
    seed: int,
) -> dict | None:
    """Compute squidpy neighbourhood enrichment of the mapped states and return a
    small summary. Needs >= 2 mapped states (returns None otherwise).

    Populates adata_st.obs[OBS_MAPPING_STATE_CAT] (categorical states), the squidpy
    spatial graph, and adata_st.uns[UNS_NHOOD_ENRICHMENT] (consumed by
    plot_nhood_enrichment). The summary reports the mean diagonal (self) and
    off-diagonal (cross) z-scores: a high self value means states preferentially
    border their own kind.
    """
    if len(np.unique(spot_states)) < 2:
        return None

    adata_st.obs[OBS_MAPPING_STATE_CAT] = pd.Categorical(spot_states.astype(str))
    spatial_connectivities(adata_st, k)  # ensure the (cached) spatial graph exists
    sq.gr.nhood_enrichment(
        adata_st,
        cluster_key=OBS_MAPPING_STATE_CAT,
        n_perms=n_perm,
        seed=seed,
        show_progress_bar=False,
    )

    zscore = np.asarray(adata_st.uns[UNS_NHOOD_ENRICHMENT]["zscore"], dtype=float)
    off = ~np.eye(zscore.shape[0], dtype=bool)
    return {
        "n_states": int(zscore.shape[0]),
        "mean_self_zscore": float(np.nanmean(np.diag(zscore))),
        "mean_cross_zscore": (
            float(np.nanmean(zscore[off])) if off.any() else float("nan")
        ),
    }
