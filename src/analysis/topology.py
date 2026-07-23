"""Spatial organisation of the mapped spots for the post-mapping analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import anndata as ad
import squidpy as sq

from adata_schema import OBS_MAPPING_HARD, OBSM_SPATIAL
from metrics.topology import local_spatial_purity, permutation_test, morans_i_mean
from plots import plot_spatial_cell_states

logger = logging.getLogger(__name__)


K_SPATIAL = 6  # neighbours for local spatial purity / Moran's I KNN graph
N_PERM_SPATIAL = 200  # permutations for the spatial-organisation null


def knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """(n_spots, k) indices of each spot's k nearest spatial neighbours (self excluded)."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(coords) - 1)
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, idx = knn.kneighbors(coords)
    return idx[:, 1:]


def spatial_neighbors_graph(coords: np.ndarray, k: int) -> ad.AnnData:
    """Minimal AnnData holding a squidpy-built spatial KNN graph, reused across permutations."""

    graph = ad.AnnData(
        X=np.zeros((len(coords), 1), dtype=np.float32),
        obsm={"spatial": np.asarray(coords, dtype=np.float32)},
    )
    sq.gr.spatial_neighbors_knn(graph, n_neighs=min(k, len(coords) - 1))
    return graph


def analyse_spatial_organization(
    adata_st: ad.AnnData,
    output_data_dir: Path,
    output_plots_dir: Path,
    state_palette: dict[int, tuple] | None = None,
    k: int = K_SPATIAL,
    n_perm: int = N_PERM_SPATIAL,
    seed: int = 0,
):
    """Local spatial purity + mean Moran's I of the mapped spot states, each with
    a permutation-null z-score. Individually undefined metrics come back as NaN;
    asserts len(coords) == len(spot_states) and len(spot_states) > k.

    Requires: adata_st.obs[OBS_MAPPING_HARD], adata_st.obsm[OBSM_SPATIAL].
    Writes topology_metrics.json under output_data_dir and
    spatial_cell_states.png under output_plots_dir.
    """
    spot_states = adata_st.obs[OBS_MAPPING_HARD].to_numpy()
    coords = np.asarray(adata_st.obsm[OBSM_SPATIAL])

    result: dict = {
        "n_spots": int(len(spot_states)),
        "n_mapped_states": int(len(np.unique(spot_states))),
        "k": int(k),
        "n_perm": int(n_perm),
        "local_purity": None,
        "morans_i": None,
    }
    assert len(coords) == len(
        spot_states
    ), "Spatial organisation skipped: coordinates length mismatch "
    assert len(spot_states) > k, "too few spots"

    rng = np.random.default_rng(seed)

    nbr_idx = knn_indices(coords, k)
    obs_lsp = local_spatial_purity(spot_states, nbr_idx)
    result["local_purity"] = permutation_test(
        obs_lsp, spot_states, lambda l: local_spatial_purity(l, nbr_idx), n_perm, rng
    )

    graph = spatial_neighbors_graph(coords, k)
    obs_mi = morans_i_mean(spot_states, graph)
    result["morans_i"] = permutation_test(
        obs_mi, spot_states, lambda l: morans_i_mean(l, graph), n_perm, rng
    )

    with open(output_data_dir / "topology_metrics.json", "w") as f:
        json.dump(result, f, indent=4)

    plot_spatial_cell_states(
        adata_st,
        spot_states,
        output_path=output_plots_dir / "spatial_cell_states.png",
        state_palette=state_palette,
    )
