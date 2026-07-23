"""Spatial-organisation metrics over spot state labels: a label-shuffling
permutation test, local spatial purity, and mean Moran's I across states."""

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
    """Shuffle ``labels`` ``n_perm`` times, recomputing ``metric_fn`` each time.

    Returns a dict with ``observed``, ``p_value`` (fraction of null >= observed),
    ``z_score``, ``null_mean`` and ``null_std``; non-finite cases yield NaNs.
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
    """Mean fraction of each spot's precomputed neighbours (``nbr_idx``) that
    share its state label."""
    return float((labels[nbr_idx] == labels[:, None]).mean())


def morans_i_mean(labels: np.ndarray, graph: ad.AnnData) -> float:
    """Moran's I averaged over each state's binary indicator, via
    ``squidpy.gr.spatial_autocorr``. Requires ``graph.obsp['spatial_connectivities']``;
    writes ``state_*`` columns onto ``graph.obs``. Returns NaN if fewer than 2 states.
    """

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
