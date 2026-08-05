"""Substate merge coherence and computed-state modularity for the post-mapping analysis."""

from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData

from adata_schema import (
    LAYER_LOGNORM,
    OBS_COMPUTED_STATE,
    OBS_LEIDEN_SHARED_GENES,
    OBS_MAPPING_HARD,
    OBSP_CONNECTIVITIES_SHARED_GENES,
    OBSP_ST_EXPR_CONNECTIVITIES,
    UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM,
    UNS_MODULARITY_SHARED_LEIDEN,
)
from metrics.biology import (
    _pairwise_cosine_stats,
    leiden_state_groups,
    compute_modularity,
)

logger = logging.getLogger(__name__)


N_PERM_COHERENCE = 200  # permutations for the substate-coherence null
SIG_ALPHA = 0.05  # p-value threshold for "significant"


def analyse_substate_coherence(
    adata_sc: AnnData,
    labels_k: np.ndarray,
    output_data_dir: Path,
    n_perm: int = N_PERM_COHERENCE,
    seed: int = 0,
):
    """Coherence of the subclusters merged into each state.

    For every state merging >=2 subclusters (with at least one subcluster in
    another state), permutation-tests the mean/median pairwise cosine similarity
    of the merged subclusters' ``centroids`` (L x G_shared) against ``n_perm``
    same-sized random draws of subclusters from other states. All-zero (empty)
    centroid rows are dropped. ``labels_k`` (L,) is the subcluster->state grouping.

    Writes biology_metrics.json under output_data_dir.
    """

    centroids = adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM]
    groups = leiden_state_groups(labels_k)
    # Only subclusters with a non-zero centroid: a zero centroid has undefined
    # cosine similarity.
    valid_leiden = {
        int(l) for l in range(len(centroids)) if np.any(centroids[l] != 0.0)
    }

    per_state: dict[int, dict] = {}
    for state, leiden_clusters in sorted(groups.items()):
        targets = [l for l in leiden_clusters if l in valid_leiden]
        others = [l for l in valid_leiden if l not in leiden_clusters]
        if len(targets) < 2:
            per_state[state] = {
                "n_leiden_sub": len(targets),
                "skipped_reason": "fewer than 2 non-empty merged Leiden clusters",
            }
            continue
        if not others:
            per_state[state] = {
                "n_leiden_sub": len(targets),
                "skipped_reason": "no Leiden clusters in other states to draw a null from",
            }
            continue

        rng = np.random.default_rng(seed + state)
        obs_mean, obs_median = _pairwise_cosine_stats(centroids[targets])

        n_draw = len(targets)
        replace = n_draw > len(others)
        others_arr = np.array(others)
        null_means = np.empty(n_perm)
        for i in range(n_perm):
            draw = rng.choice(others_arr, size=n_draw, replace=replace)
            null_means[i], _ = _pairwise_cosine_stats(centroids[draw])

        p_mean = float((null_means >= obs_mean).mean())
        z_mean = float((obs_mean - null_means.mean()) / (null_means.std() + 1e-12))
        per_state[state] = {
            "n_leiden_sub": len(targets),
            "mean_cossim": obs_mean,
            "median_cossim": obs_median,
            "null_mean": float(null_means.mean()),
            "p_value_mean": p_mean,
            "z_score_mean": z_mean,
            "skipped_reason": None,
        }

    tested = [m for m in per_state.values() if m.get("skipped_reason") is None]
    if tested:
        aggregate = {
            "n_tested_states": len(tested),
            "mean_cossim": float(np.mean([m["mean_cossim"] for m in tested])),
            "mean_z_score": float(np.mean([m["z_score_mean"] for m in tested])),
            "frac_significant": float(
                np.mean([m["p_value_mean"] < SIG_ALPHA for m in tested])
            ),
        }
    else:
        aggregate = {
            "n_tested_states": 0,
            "mean_cossim": float("nan"),
            "mean_z_score": float("nan"),
            "frac_significant": float("nan"),
        }

    with open(output_data_dir / "biology_metrics.json", "w") as f:
        json.dump(
            {"n_perm": int(n_perm), "aggregate": aggregate, "per_state": per_state},
            f,
            indent=4,
        )


def _spatial_expression_modularity(adata_st: AnnData) -> float:
    """Modularity of the hard spot->state labels on the ST expression KNN graph.

    Measures transcriptional coherence of the mapping within the spatial data:
    builds an expression KNN graph over all genes of ``adata_st`` (PCA of the
    lognorm layer + scanpy neighbors, matching the reference all-genes graph)
    and computes the modularity of ``adata_st.obs[OBS_MAPPING_HARD]`` on it.

    Requires: adata_st.layers[LAYER_LOGNORM], adata_st.obs[OBS_MAPPING_HARD].
    """
    # The expression KNN graph is K-independent (only the hard labels change per
    # K), so build it once and cache the connectivities on adata_st.
    if OBSP_ST_EXPR_CONNECTIVITIES not in adata_st.obsp:
        # Throwaway AnnData so the PCA / neighbor graph never touches adata_st.
        st_expr = AnnData(
            X=adata_st.layers[LAYER_LOGNORM].copy(),
            obs=adata_st.obs[[]].copy(),
            var=adata_st.var[[]].copy(),
        )
        n = min(30, st_expr.n_obs - 1, st_expr.n_vars - 1)
        sc.pp.pca(st_expr, n_comps=n)
        sc.pp.neighbors(st_expr)
        adata_st.obsp[OBSP_ST_EXPR_CONNECTIVITIES] = st_expr.obsp["connectivities"]

    graph = AnnData(
        X=np.zeros((adata_st.n_obs, 1), dtype=np.float32),
        obsp={"connectivities": adata_st.obsp[OBSP_ST_EXPR_CONNECTIVITIES]},
    )
    hard_labels = adata_st.obs[OBS_MAPPING_HARD].astype(int).to_numpy()
    return compute_modularity(graph, hard_labels)


def analyse_modularities(
    adata_sc: AnnData,
    adata_st: AnnData,
    output_data_dir: Path,
):
    """Modularity of the computed-state partition on the reference KNN graphs.

    Also measures transcriptional coherence of the mapping within the spatial
    data via ``_spatial_expression_modularity`` (hard spot->state labels on the
    ST expression KNN graph over all genes).

    Requires: adata_sc.obs[OBS_COMPUTED_STATE], adata_sc.obs[OBS_LEIDEN_SHARED_GENES],
        adata_sc.obsp[OBSP_CONNECTIVITIES_SHARED_GENES],
        adata_sc.uns[UNS_LEIDEN_RESOLUTION_ALL_GENES],
        adata_st.layers[LAYER_LOGNORM], adata_st.obs[OBS_MAPPING_HARD].
    Writes modularity_metrics.json under output_data_dir.
    """

    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()

    modularity_all = compute_modularity(adata_sc, cell_states)
    modularity_shared = compute_modularity(
        adata_sc, cell_states, obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES
    )
    # The shared-gene Leiden partition does not depend on K, so compute its
    # modularity once and cache it on adata_sc.
    if UNS_MODULARITY_SHARED_LEIDEN not in adata_sc.uns:
        adata_sc.uns[UNS_MODULARITY_SHARED_LEIDEN] = compute_modularity(
            adata_sc,
            adata_sc.obs[OBS_LEIDEN_SHARED_GENES].astype(int).to_numpy(),
            obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES,
        )
    modularity_shared_leiden = adata_sc.uns[UNS_MODULARITY_SHARED_LEIDEN]
    modularity_st_expression = _spatial_expression_modularity(adata_st)

    with open(output_data_dir / "modularity_metrics.json", "w") as f:
        json.dump(
            {
                "modularity_all": float(modularity_all),
                "modularity_shared": float(modularity_shared),
                "modularity_shared_leiden": float(modularity_shared_leiden),
                "modularity_st_expression": float(modularity_st_expression),
            },
            f,
            indent=4,
        )
