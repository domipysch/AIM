"""Agglomeration tree over the Leiden-subcluster centroids: build the linkage once,
then cut it at any K to get subcluster->state labels."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from aim.aim_config import AGGLO_TREE_METHODS


def build_agglomeration_tree(
    centroids: np.ndarray,
    method: str = AGGLO_TREE_METHODS[0],
    eps: float = 1e-8,
) -> np.ndarray:
    """Linkage matrix over the subcluster centroids, following the tree construction of
    Grabski et al. (2023) — the ``testClusters`` routine of their sc-SHC reference
    implementation (R/clustering.R):

    1. rescale each subcluster to relative gene frequencies (rows sum to 1);
    2. Euclidean distance between those profiles;
    3. agglomerate with ``method``.

    Step 1 pseudobulks by *summing* counts per cluster and dividing by the cluster's
    total, which normalizes sequencing depth out. Passing
    ``UNS_LEIDEN_CENTROIDS_SHARED_GENES`` (mean counts per cell) gives an identical
    result, because the per-cell divisor cancels under the row normalization:
    ``(S/n) / sum(S/n) == S / sum(S)``. Passing ``UNS_LEIDEN_EXPR_SUMS_SHARED_GENES``
    is equally valid. Note this is *not* the "average expression" the paper's Methods
    text describes — the row totals differ, and only the normalized form is
    depth-invariant.

    No residual or GLM-PCA transform is applied: the paper's argument is that with
    many cells per cluster these centers are no longer small counts. GLM-PCA enters
    their pipeline only inside the per-node significance test, on cells, which is not
    implemented here. Their upstream feature selection (the 2,500 highest-deviance
    genes) is also omitted — the tree is built on whatever genes ``centroids``
    carries, i.e. all shared genes in the AIM sweep.

    ``method`` is one of ``AGGLO_TREE_METHODS`` (``aim_config``), i.e. the same set
    the ``--agglo_tree_method`` CLI flag and the GUI sidebar offer:

    - ``"ward"`` (default): Ward's criterion, reproducing the paper's
      ``hclust(dist(...), method="ward.D")``. scipy's ``method="ward"`` is R's
      *ward.D2* (it squares the input internally), so the unsquared Euclidean
      distances are passed as ``sqrt(d)`` to recover ward.D, using the identity
      ``ward.D(d) == ward.D2(sqrt(d))``. Ward carries a size term and tends to
      produce balanced states.
    - ``"average"``: UPGMA — average pairwise distance. No squaring convention
      applies (average linkage has no ward.D/ward.D2 distinction and does not square
      internally), so it runs on the plain Euclidean distances ``d``; this also keeps
      both methods operating on the same ``d``, so they are directly comparable.
      Average linkage has no size term and tends to peel small tight groups off a
      growing dominant state.

    Every leaf is weighted equally regardless of how many cells it pools, so this is
    agglomeration over the profiles, not over the underlying cells. Rows that are zero
    across every gene stay zero; the all-zero centroid rows nudged to a uniform
    ``1e-6`` by ``compute_leiden_aggregates`` become a uniform composition here.
    """
    if method not in AGGLO_TREE_METHODS:
        raise ValueError(f"method must be one of {AGGLO_TREE_METHODS}, got {method!r}")

    profiles = np.asarray(centroids, dtype=np.float64)
    if profiles.shape[0] < 2:
        raise ValueError(
            f"need at least 2 subclusters to build a tree, got {profiles.shape[0]}"
        )
    if not np.isfinite(profiles).all():
        raise ValueError("centroids contain non-finite values")

    profiles = profiles / (profiles.sum(axis=1, keepdims=True) + eps)

    distances = pdist(profiles, metric="euclidean")
    if method == "ward":
        distances = np.sqrt(distances)
    return linkage(distances, method=method)


def labels_at_k(linkage_z: np.ndarray, k: int, n_leiden: int) -> np.ndarray:
    """Cut the linkage tree into k clusters; returns a subcluster->state label array (0..k-1)."""
    if k >= n_leiden:
        return np.arange(n_leiden, dtype=int)
    raw = fcluster(linkage_z, t=k, criterion="maxclust")
    _, remapped = np.unique(raw, return_inverse=True)
    return remapped.astype(int)
