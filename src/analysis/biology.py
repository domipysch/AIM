"""Biological / topological metrics for the post-mapping analysis.

Three questions, each producing a scalar objective (every permutation-tested
metric is reported with an observed value AND a null z-score; the z-score is the
dataset-comparable objective, since raw purity / Moran's I / cosine values are
not comparable across pairs with different numbers of states):

1. Spatial organisation of the mapped spots (P's argmax state per spot)
   - Local Spatial Purity (LSP): mean fraction of the k nearest spatial
     neighbours sharing the same mapped state — local compactness.
   - Moran's I averaged over the per-state binary indicators — global spatial
     clustering (via squidpy).
   - Null: shuffle the spot-state labels N_PERM times.

2. Coherence of the subclusters merged into one computed state
   - For each computed state that aggregates >=2 subclusters: are those
     subclusters' shared-gene centroids MORE mutually similar (higher mean
     pairwise cosine) than a same-sized random draw of subclusters from OTHER
     states? If yes (high z-score, low p), the merge looks coherent in the gene
     subspace the method actually operates in.
   - Null: N_PERM random same-sized draws of subclusters from other states.

3. Modularity of the computed-state partition on a precomputed KNN graph.

These are generic given their inputs (label arrays, centroids, a graph), so they
live in ``metrics`` rather than in the AIM-specific ``analysis`` package.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np
from anndata import AnnData

from adata_schema import (
    OBS_COMPUTED_STATE,
    OBS_LEIDEN_SHARED_GENES,
    OBSP_CONNECTIVITIES_SHARED_GENES,
    UNS_LEIDEN_RESOLUTION_ALL_GENES,
)
from metrics.biology import (
    _pairwise_cosine_stats,
    leiden_state_groups,
    compute_modularity,
)
from plots import plot_umap_comparison, plot_umap_grid

logger = logging.getLogger(__name__)


N_PERM_COHERENCE = 200  # permutations for the substate-coherence null
SIG_ALPHA = 0.05  # p-value threshold for "significant" counts


def analyse_substate_coherence(
    centroids: np.ndarray,
    labels_k: np.ndarray,
    output_data_dir: Path,
    n_perm: int = N_PERM_COHERENCE,
    seed: int = 0,
):
    """Shared-gene centroid coherence of the subclusters merged into each state.

    For every computed state that aggregates >=2 subclusters (and with at least
    one subcluster belonging to another state, to draw a null from): compute the
    mean/median pairwise cosine similarity of the merged subclusters' shared-gene
    centroids, and permutation-test it against ``n_perm`` random same-sized draws
    of subclusters from OTHER states.

    Args:
        centroids: per-subcluster shared-gene centroid (L x G_shared), e.g.
                   normalized+log1p expression sums / sizes. Rows for empty
                   subclusters (size 0) may be all-zero; such rows are dropped.
        labels_k:  subcluster -> state label array (L,); defines the grouping.

    Returns a dict with per-state detail and aggregate scalars.
    """

    groups = leiden_state_groups(labels_k)
    # Subclusters that actually have cells (non-zero centroid) — a zero centroid
    # has undefined cosine similarity and must not seed a null draw.
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

    # Save to file
    with open(output_data_dir / "topology_metrics.json", "w") as f:
        json.dump(
            {"n_perm": int(n_perm), "aggregate": aggregate, "per_state": per_state},
            f,
            indent=4,
        )


def analyse_modularities(
    adata_sc: AnnData,
    output_data_dir: Path,
    output_plots_dir: Path,
    state_palette: dict[int, tuple] | None = None,
):

    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()

    modularity_all = compute_modularity(adata_sc, cell_states)
    modularity_shared = compute_modularity(
        adata_sc, cell_states, obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES
    )
    modularity_shared_leiden = compute_modularity(
        adata_sc,
        adata_sc.obs[OBS_LEIDEN_SHARED_GENES].astype(int).to_numpy(),
        obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES,
    )
    logger.info(
        "Modularity: all=%.4f shared=%.4f | shared Leiden ref=%.4f",
        modularity_all,
        modularity_shared,
        modularity_shared_leiden,
    )

    with open(output_data_dir / "modularity_metrics.json", "w") as f:
        json.dump(
            {
                "modularity_all": float(modularity_all),
                "modularity_shared": float(modularity_shared),
                "modularity_shared_leiden": float(modularity_shared_leiden),
            },
            f,
            indent=4,
        )

    # ── Computed-state UMAP (standalone, for side-by-side with the spatial plot)
    plot_umap_comparison(
        adata_sc,
        panels=[(OBS_COMPUTED_STATE, "Computed cell-state assignment")],
        output_path=output_plots_dir / "umap_computed_state.png",
        state_palette=state_palette,
    )

    # ── UMAP grid: {Leiden all-gene, Leiden shared-gene, computed} x {all, shared}
    plot_umap_grid(
        adata_sc,
        output_path=output_plots_dir / "umap_grid.png",
        leiden_resolution=float(adata_sc.uns[UNS_LEIDEN_RESOLUTION_ALL_GENES]),
        state_palette=state_palette,
        modularity_all=modularity_all,
        modularity_shared=modularity_shared,
    )
