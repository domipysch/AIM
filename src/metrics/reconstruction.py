"""Reconstruct predicted spot expression from a mapping and per-state centroids
(``mapping @ state_centroids``), and assemble those centroids from subcluster
profiles."""

from __future__ import annotations

import numpy as np


def assemble_state_centroids(
    leiden_to_state: np.ndarray,
    k: int,
    expr_sums: np.ndarray,
    sizes: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Assemble per-state centroids (k x G) as, for each state, the summed
    subcluster expression divided by the summed subcluster sizes.

    ``leiden_to_state`` (L,) maps each subcluster to a state 0..k-1, ``expr_sums``
    is summed expression per subcluster (L x G), ``sizes`` is cells per subcluster
    (L,), and ``eps`` guards the denominator for states with no support.
    """
    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=expr_sums.dtype)
    np.add.at(state_sums, leiden_to_state, expr_sums)
    state_sizes = np.zeros(k, dtype=sizes.dtype)
    np.add.at(state_sizes, leiden_to_state, sizes)
    return state_sums / (state_sizes[:, None] + eps)


def predict_expression(mapping, centroids) -> np.ndarray:
    """Predicted spot expression: ``mapping`` (S x k) @ ``centroids`` (k x G),
    returning S x G. Both inputs are coerced with ``np.asarray``."""
    return np.asarray(mapping) @ np.asarray(centroids)
