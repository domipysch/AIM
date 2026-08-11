"""Reconstruct predicted spot expression from a mapping and per-state centroids
(``mapping @ state_centroids``), assemble those centroids from subcluster
profiles, and score the reconstruction against a label-shuffling null."""

from __future__ import annotations

import numpy as np

from .cossim import cosine_along_axis


def assemble_state_centroids(
    start_cluster_to_state: np.ndarray,
    k: int,
    expr_sums: np.ndarray,
    sizes: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Assemble per-state centroids (k x G) as, for each state, the summed
    subcluster expression divided by the summed subcluster sizes.

    ``start_cluster_to_state`` (L,) maps each start cluster to a state 0..k-1, ``expr_sums``
    is summed expression per subcluster (L x G), ``sizes`` is cells per subcluster
    (L,), and ``eps`` guards the denominator for states with no support.
    """
    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=expr_sums.dtype)
    np.add.at(state_sums, start_cluster_to_state, expr_sums)
    state_sizes = np.zeros(k, dtype=sizes.dtype)
    np.add.at(state_sizes, start_cluster_to_state, sizes)
    return state_sums / (state_sizes[:, None] + eps)


def predict_expression(mapping, centroids) -> np.ndarray:
    """Predicted spot expression: ``mapping`` (S x k) @ ``centroids`` (k x G),
    returning S x G. Both inputs are coerced with ``np.asarray``."""
    return np.asarray(mapping) @ np.asarray(centroids)


def cossim_null_medians(
    measured: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Reconstruction cosine similarity at chance level: shuffle the labels.

    Each of ``n_perm`` shuffles reassigns the spot->state labels at random (the
    per-state spot counts are preserved, only *which* spot gets which state
    changes), reconstructs ``centroids[shuffled]`` and reduces it exactly like the
    observed value does — median over spots and median over genes of the per-spot
    / per-gene cosine against ``measured`` (S x G, same gene order as
    ``centroids``). Returns the mean of those medians as
    ``{"median_spot", "median_gene"}``; NaN for ``n_perm <= 0``.
    """
    measured = np.asarray(measured)
    labels = np.asarray(labels)
    per_spot, per_gene = [], []
    for _ in range(n_perm):
        predicted = np.asarray(centroids)[rng.permutation(labels)]
        per_spot.append(
            float(np.median(cosine_along_axis(measured, predicted, axis=1)))
        )
        per_gene.append(
            float(np.median(cosine_along_axis(measured, predicted, axis=0)))
        )
    if not per_spot:
        return {"median_spot": float("nan"), "median_gene": float("nan")}
    return {
        "median_spot": float(np.mean(per_spot)),
        "median_gene": float(np.mean(per_gene)),
    }
