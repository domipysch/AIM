"""Agglomeration tree over the Leiden-subcluster centroids: build the linkage once,
then cut it at any K to get subcluster->state labels."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def build_agglomeration_tree(
    centroids: np.ndarray, method: str, metric: str
) -> np.ndarray:
    """Average-linkage linkage matrix over centroids using cosine distance."""
    return linkage(centroids, method=method, metric=metric)


def labels_at_k(linkage_z: np.ndarray, k: int, n_leiden: int) -> np.ndarray:
    """Cut the linkage tree into k clusters; returns a subcluster->state label array (0..k-1)."""
    if k >= n_leiden:
        return np.arange(n_leiden, dtype=int)
    raw = fcluster(linkage_z, t=k, criterion="maxclust")
    _, remapped = np.unique(raw, return_inverse=True)
    return remapped.astype(int)
