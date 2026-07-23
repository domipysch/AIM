"""Expression reconstruction from a mapping and per-state centroids.

Both aligner families in this project reconstruct predicted spot expression the
same way — ``mapping @ state_centroids`` — so the math lives here once and is
shared by ``analysis`` (AIM) and ``reference_aligners/mapping_analysis``
(Tangram/TACCO/DOT).

``assemble_state_centroids`` is the numpy assembly of per-state centroids from a
subcluster->state label array plus fixed per-cluster expression sums / sizes. It
is the post-hoc mirror of ``aim.aggregation.assemble_state_profiles_shared_genes``
(which does the same on torch, on shared-gene raw counts only, inside the sweep);
this version is numpy and runs on whichever expression sums it is handed (raw or
normalized), for the disk-based reconstruction scoring.
"""

from __future__ import annotations

import numpy as np


def assemble_state_centroids(
    leiden_to_state: np.ndarray,
    k: int,
    expr_sums: np.ndarray,
    sizes: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Assemble per-state gene expression centroids from a subcluster->state label
    array and the fixed per-cluster expression sums / sizes:

        M[s] = (sum_{l: leiden_to_state[l]=s} expr_sums[l]) / (sum_{l: leiden_to_state[l]=s} sizes[l])

    Args:
        leiden_to_state: subcluster -> state label array (L,), values 0..k-1.
        k: number of states (rows of the returned M).
        expr_sums: summed expression per subcluster (L x G).
        sizes: number of cells per subcluster (L,).
        eps: added to the denominator to avoid division by zero for states with
             no subcluster support.

    Returns:
        M: state gene expression profiles (k x G).
    """
    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=expr_sums.dtype)
    np.add.at(state_sums, leiden_to_state, expr_sums)
    state_sizes = np.zeros(k, dtype=sizes.dtype)
    np.add.at(state_sizes, leiden_to_state, sizes)
    return state_sums / (state_sizes[:, None] + eps)


def predict_expression(mapping, centroids) -> np.ndarray:
    """Predicted spot expression Z' = mapping @ centroids (S x G).

    ``mapping`` (S x k) and ``centroids`` (k x G) may be any array-like (ndarray
    or DataFrame); both are coerced with ``np.asarray`` first.
    """
    return np.asarray(mapping) @ np.asarray(centroids)
