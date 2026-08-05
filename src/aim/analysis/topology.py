"""Spatial organisation of the mapped spots for the post-mapping analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import anndata as ad
import squidpy as sq

from aim.adata_schema import (
    OBS_MAPPING_HARD,
    OBSM_SPATIAL,
    OBSP_SPATIAL_CONNECTIVITIES,
)
from aim.metrics.topology import local_spatial_purity, permutation_test

logger = logging.getLogger(__name__)


K_SPATIAL = 6  # neighbours for the squidpy spatial KNN graph
N_PERM_SPATIAL = 100  # permutations for the spatial-organisation null


def spatial_connectivities(adata_st: ad.AnnData, k: int):
    """Squidpy spatial KNN connectivity matrix, built once and cached on adata_st.

    Depends only on the spot coordinates, which are constant across the K-sweep,
    so it is computed on the first K and reused for every later K. Serves both the
    local spatial purity metric and the neighbourhood enrichment. Returns the
    (n_spots x n_spots) sparse adjacency in adata_st.obsp[OBSP_SPATIAL_CONNECTIVITIES].
    """
    if OBSP_SPATIAL_CONNECTIVITIES not in adata_st.obsp:
        sq.gr.spatial_neighbors_knn(adata_st, n_neighs=k)
    return adata_st.obsp[OBSP_SPATIAL_CONNECTIVITIES]


def analyse_spatial_organization(
    adata_st: ad.AnnData,
    output_data_dir: Path,
    k: int = K_SPATIAL,
    n_perm: int = N_PERM_SPATIAL,
    seed: int = 0,
):
    """Local spatial purity + self neighbourhood enrichment of the mapped spot states.

    Local spatial purity comes with a permutation-null z-score; the self
    neighbourhood-enrichment z-score summarises how much each state preferentially
    borders itself (see ``_nhood_enrichment``). Individually undefined metrics come
    back as NaN/None; asserts len(coords) == len(spot_states) and
    len(spot_states) > k.

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
    """Self neighbourhood-enrichment z-score of the mapped states, averaged over
    states. Needs >= 2 mapped states (returns None otherwise).

    Only the *self* term is computed -- the diagonal of squidpy's neighbourhood
    enrichment -- since ``mean_self_zscore`` is the sole value consumed downstream.
    For each state ``s`` we count the spatial-graph edges whose two endpoints are
    both in ``s`` and z-score that against a label-shuffling null (the same null
    ``sq.gr.nhood_enrichment`` uses), then average the per-state z-scores. This
    skips squidpy's full K x K enrichment matrix and its per-K call overhead, so it
    is markedly cheaper than a full ``nhood_enrichment`` run. A high value means
    states preferentially border their own kind.

    Values are comparable to (not bit-identical with) squidpy's diagonal: same
    graph and null, different permutation RNG.
    """
    if len(np.unique(spot_states)) < 2:
        return None

    A = spatial_connectivities(adata_st, k).tocoo()
    row, col = A.row, A.col
    n_states = int(spot_states.max()) + 1

    def within_state_edges(labels: np.ndarray) -> np.ndarray:
        """Per-state count of spatial edges whose endpoints share that state."""
        same = labels[row] == labels[col]
        return np.bincount(labels[row][same], minlength=n_states).astype(np.float64)

    observed = within_state_edges(spot_states)
    rng = np.random.default_rng(seed)
    null = np.stack(
        [within_state_edges(rng.permutation(spot_states)) for _ in range(n_perm)]
    )
    mean = null.mean(axis=0)
    std = null.std(axis=0)

    present = np.unique(spot_states)
    z = np.full(n_states, np.nan)
    ok = std > 0
    z[ok] = (observed[ok] - mean[ok]) / std[ok]
    self_z = z[present]
    return {
        "n_states": int(len(present)),
        "mean_self_zscore": (
            float(np.nanmean(self_z)) if np.isfinite(self_z).any() else float("nan")
        ),
    }
