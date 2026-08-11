"""Start clusters for the AIM sweep: the partition the agglomeration tree is built
over, plus the reference neighbor graphs the analysis measures against.

Two ways to get start clusters:

* **Leiden over-clustering** (the default) -- ``run_leiden_clustering_all_genes``
  over-clusters the reference on all genes.
* **A pre-existing annotation** -- ``set_start_clusters_from_annotation`` takes an
  ``obs`` column of cell-type labels as the starting partition, so no Leiden
  over-clustering is computed at all.

Both write the same keys (``OBS_START_CLUSTER`` / ``UNS_N_START_CLUSTERS`` /
``UNS_START_CLUSTER_NAMES``), so everything downstream is identical.

Building the neighbor graphs is split out from running Leiden on them
(``build_neighbor_graph_*``): the graphs are needed in both modes -- the modularity
metrics measure a partition against them and the UMAPs are laid out from them --
while the all-gene Leiden itself is only run in the first mode. The shared-gene
Leiden is a property of the reference (it feeds ``modularity_shared_leiden``), not a
starting partition, so it runs in both modes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from aim.adata_schema import (
    LAYER_LOGNORM,
    OBS_LEIDEN_SHARED_GENES,
    OBS_START_CLUSTER,
    OBSM_LOGNORM_SHARED_GENES,
    OBSM_PCA,
    OBSM_PCA_SHARED_GENES,
    OBSP_CONNECTIVITIES_SHARED_GENES,
    OBSP_DISTANCES_SHARED_GENES,
    UNS_LEIDEN_NUMBER_STATES_SHARED_GENES,
    UNS_LEIDEN_RESOLUTION_ALL_GENES,
    UNS_LEIDEN_RESOLUTION_SHARED_GENES,
    UNS_N_START_CLUSTERS,
    UNS_NEIGHBORS_SHARED_GENES,
    UNS_SHARED_GENES,
    UNS_START_CLUSTER_NAMES,
)

logger = logging.getLogger(__name__)


def build_neighbor_graph_all_genes(
    adata_sc: AnnData,
    n_pca_comps: int = 30,
) -> None:
    """
    PCA + KNN graph of the reference over all genes, in place; ``adata_sc.X`` is left raw.

    Needed in both start-cluster modes: the all-gene Leiden runs on this graph, the
    ``modularity_all`` metric measures the computed states against it, and the
    all-gene UMAP is laid out from it.

    Requires: adata_sc.layers[LAYER_LOGNORM].
    Adds: adata_sc.obsm[OBSM_PCA] (plus the scanpy neighbor graph under its default keys).
    """
    n = min(n_pca_comps, adata_sc.n_obs - 1, adata_sc.n_vars - 1)
    sc.pp.pca(adata_sc, n_comps=n, layer=LAYER_LOGNORM, key_added=OBSM_PCA)
    sc.pp.neighbors(adata_sc, use_rep=OBSM_PCA)


def run_leiden_clustering_all_genes(
    adata_sc: AnnData,
    resolution: float,
    n_pca_comps: int = 30,
):
    """
    Leiden over-cluster the reference on all genes into start clusters, in place.

    Requires: adata_sc.layers[LAYER_LOGNORM].
    Adds: adata_sc.obsm[OBSM_PCA], adata_sc.obs[OBS_START_CLUSTER],
    adata_sc.uns[UNS_START_CLUSTER_NAMES], adata_sc.uns[UNS_N_START_CLUSTERS],
    adata_sc.uns[UNS_LEIDEN_RESOLUTION_ALL_GENES] (plus the scanpy neighbor graph).
    """

    build_neighbor_graph_all_genes(adata_sc, n_pca_comps=n_pca_comps)

    sc.tl.leiden(
        adata_sc,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        key_added=OBS_START_CLUSTER,
    )
    number_of_clusters = len(
        np.unique(adata_sc.obs[OBS_START_CLUSTER].astype(int).values)
    )
    logger.info(
        "Leiden clustering (resolution=%.2f): %d start clusters",
        resolution,
        number_of_clusters,
    )

    adata_sc.uns[UNS_LEIDEN_RESOLUTION_ALL_GENES] = resolution
    adata_sc.uns[UNS_N_START_CLUSTERS] = number_of_clusters
    adata_sc.uns[UNS_START_CLUSTER_NAMES] = [
        f"cluster_{i}" for i in range(number_of_clusters)
    ]


def drop_unannotated(adata_sc: AnnData, cell_type_key: str) -> AnnData:
    """Return ``adata_sc`` without the cells that carry no ``cell_type_key`` label.

    A cell with no label cannot be a start cluster; dropping it beats letting it
    become a spurious "nan" type. Returns the same object when nothing is dropped,
    a copy otherwise -- so callers must rebind. Raises ``KeyError`` (listing the
    available columns) when the annotation column is missing.
    """
    if cell_type_key not in adata_sc.obs:
        raise KeyError(
            f"cell_type_key {cell_type_key!r} not in obs; "
            f"available: {list(adata_sc.obs.columns)}"
        )
    labelled = adata_sc.obs[cell_type_key].notna().to_numpy()
    if labelled.all():
        return adata_sc
    logger.warning(
        "Dropping %d of %d cell(s) with no %r label",
        int((~labelled).sum()),
        adata_sc.n_obs,
        cell_type_key,
    )
    return adata_sc[labelled].copy()


def set_start_clusters_from_annotation(
    adata_sc: AnnData, cell_type_key: str
) -> list[str]:
    """Use the annotation in ``obs[cell_type_key]`` as the start clusters, in place.

    The annotated types take the place of the Leiden over-clustering: their
    categorical codes (contiguous 0..L-1, in ``categories`` order) become the start
    cluster of every cell, so no Leiden over-clustering is computed. Run
    :func:`drop_unannotated` first -- an unlabelled cell would get code -1 here.

    Adds: adata_sc.obs[OBS_START_CLUSTER], adata_sc.uns[UNS_N_START_CLUSTERS],
    adata_sc.uns[UNS_START_CLUSTER_NAMES]. Returns the type names.
    """
    types = pd.Categorical(adata_sc.obs[cell_type_key].astype(str))
    names = [str(c) for c in types.categories]

    adata_sc.obs[OBS_START_CLUSTER] = np.asarray(types.codes, dtype=np.int64)
    adata_sc.uns[UNS_N_START_CLUSTERS] = len(names)
    adata_sc.uns[UNS_START_CLUSTER_NAMES] = names
    logger.info(
        "Start clusters from annotation %r: %d type(s)", cell_type_key, len(names)
    )
    return names


def build_neighbor_graph_shared_genes(
    adata_sc: AnnData,
    n_pca_comps: int = 30,
) -> None:
    """
    PCA + KNN graph of the reference restricted to the shared genes, in place on
    ``adata_sc``. PCA and neighbors run inside a throwaway shared-gene AnnData, then
    the results are copied back under the shared-gene keys.

    Requires: adata_sc.uns[UNS_SHARED_GENES], adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES].
    Adds: adata_sc.obsm[OBSM_PCA_SHARED_GENES], adata_sc.uns[UNS_NEIGHBORS_SHARED_GENES],
    adata_sc.obsp[OBSP_DISTANCES_SHARED_GENES], adata_sc.obsp[OBSP_CONNECTIVITIES_SHARED_GENES].
    """
    shared_genes = adata_sc.uns[UNS_SHARED_GENES]
    adata_shared = AnnData(
        X=adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES],
        obs=adata_sc.obs[[]].copy(),
        var=pd.DataFrame(index=shared_genes),
    )

    n = min(n_pca_comps, adata_shared.n_obs - 1, adata_shared.n_vars - 1)
    sc.pp.pca(adata_shared, n_comps=n, key_added=OBSM_PCA_SHARED_GENES)

    sc.pp.neighbors(
        adata_shared,
        use_rep=OBSM_PCA_SHARED_GENES,
        key_added=UNS_NEIGHBORS_SHARED_GENES,
    )

    # Copy the shared-gene PCA and neighbor graph back onto adata_sc. The copied
    # uns entry carries the connectivities/distances key names, so scanpy tools
    # (leiden, umap) can run on adata_sc with neighbors_key=UNS_NEIGHBORS_SHARED_GENES.
    adata_sc.obsm[OBSM_PCA_SHARED_GENES] = adata_shared.obsm[OBSM_PCA_SHARED_GENES]
    adata_sc.uns[UNS_NEIGHBORS_SHARED_GENES] = adata_shared.uns[
        UNS_NEIGHBORS_SHARED_GENES
    ]
    adata_sc.obsp[OBSP_DISTANCES_SHARED_GENES] = adata_shared.obsp[
        OBSP_DISTANCES_SHARED_GENES
    ]
    adata_sc.obsp[OBSP_CONNECTIVITIES_SHARED_GENES] = adata_shared.obsp[
        OBSP_CONNECTIVITIES_SHARED_GENES
    ]


def run_leiden_clustering_shared_genes(
    adata_sc: AnnData,
    resolution: float,
    n_pca_comps: int = 30,
):
    """
    Leiden-cluster the reference restricted to the shared genes, in place on ``adata_sc``.

    This is a property of the reference, not a starting partition: it is what
    ``modularity_shared_leiden`` measures and what the shared-gene UMAP is coloured
    by, so it runs whichever way the start clusters were obtained.

    Requires: adata_sc.uns[UNS_SHARED_GENES], adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES].
    Adds: everything :func:`build_neighbor_graph_shared_genes` adds, plus
    adata_sc.obs[OBS_LEIDEN_SHARED_GENES], adata_sc.uns[UNS_LEIDEN_RESOLUTION_SHARED_GENES],
    adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_SHARED_GENES].
    """
    build_neighbor_graph_shared_genes(adata_sc, n_pca_comps=n_pca_comps)

    sc.tl.leiden(
        adata_sc,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBS_LEIDEN_SHARED_GENES,
    )
    number_of_clusters = len(
        np.unique(adata_sc.obs[OBS_LEIDEN_SHARED_GENES].astype(int).values)
    )
    logger.info(
        "Leiden clustering on shared genes (resolution=%.2f): %d clusters",
        resolution,
        number_of_clusters,
    )

    adata_sc.uns[UNS_LEIDEN_RESOLUTION_SHARED_GENES] = resolution
    adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_SHARED_GENES] = number_of_clusters
