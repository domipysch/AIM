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
from typing import Callable

import anndata as ad
import numpy as np
import squidpy as sq

logger = logging.getLogger(__name__)


def permutation_test(
    observed: float,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    """Shuffle ``labels`` n_perm times, recompute ``metric_fn`` each time.

    Returns observed value plus p-value (fraction of null >= observed), z-score,
    and the null mean/std. Full null draws are not retained (keeps the JSON small).
    """
    if not np.isfinite(observed):
        return {
            "observed": float(observed),
            "p_value": float("nan"),
            "z_score": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
        }
    perm = np.array([metric_fn(rng.permutation(labels)) for _ in range(n_perm)])
    valid = perm[np.isfinite(perm)]
    if len(valid) == 0:
        return {
            "observed": float(observed),
            "p_value": float("nan"),
            "z_score": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
        }
    return {
        "observed": float(observed),
        "p_value": float((valid >= observed).mean()),
        "z_score": float((observed - valid.mean()) / (valid.std() + 1e-12)),
        "null_mean": float(valid.mean()),
        "null_std": float(valid.std()),
    }


def local_spatial_purity(labels: np.ndarray, nbr_idx: np.ndarray) -> float:
    """Mean fraction of k nearest spatial neighbours sharing the same state label,
    using precomputed neighbour indices (so the permutation null is cheap)."""
    return float((labels[nbr_idx] == labels[:, None]).mean())


def morans_i_mean(labels: np.ndarray, graph: ad.AnnData) -> float:
    """Moran's I averaged over all states' binary indicators, via squidpy.gr.spatial_autocorr."""

    states = np.unique(labels)
    if len(states) < 2:
        return float("nan")
    state_cols = [f"state_{s}" for s in states]
    onehot = np.zeros((len(labels), len(state_cols)), dtype=np.float32)
    for col, s in enumerate(states):
        onehot[:, col] = labels == s
    graph.obs[state_cols] = onehot

    result = sq.gr.spatial_autocorr(
        graph,
        mode="moran",
        genes=state_cols,
        attr="obs",
        connectivity_key="spatial_connectivities",
        n_perms=None,
        n_jobs=1,
        show_progress_bar=False,
        copy=True,
    )
    values = result["I"].to_numpy()
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else float("nan")
