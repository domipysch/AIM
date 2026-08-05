"""Spatial-organisation metrics over spot state labels: a label-shuffling
permutation test and local spatial purity."""

from __future__ import annotations

import logging
from typing import Callable
import numpy as np

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


def local_spatial_purity(labels: np.ndarray, connectivities) -> float:
    """Fraction of neighbour pairs in the spatial graph that share a state label.

    ``connectivities`` is a (n_spots x n_spots) sparse adjacency (e.g. squidpy's
    ``spatial_connectivities``); each stored edge is counted once, ignoring its
    weight. Returns NaN for an empty graph.
    """
    A = connectivities.tocoo()
    if A.nnz == 0:
        return float("nan")
    return float((labels[A.row] == labels[A.col]).mean())
