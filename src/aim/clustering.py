"""Leiden over-clustering of the scRNA reference for the AIM sweep."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from adata_schema import (
    LAYER_LOGNORM,
    OBS_LEIDEN_ALL_GENES,
    OBS_LEIDEN_SHARED_GENES,
    OBSM_LOGNORM_SHARED_GENES,
    OBSM_PCA,
    OBSM_PCA_SHARED_GENES,
    OBSP_CONNECTIVITIES_SHARED_GENES,
    OBSP_DISTANCES_SHARED_GENES,
    UNS_LEIDEN_NUMBER_STATES_ALL_GENES,
    UNS_LEIDEN_NUMBER_STATES_SHARED_GENES,
    UNS_LEIDEN_RESOLUTION_ALL_GENES,
    UNS_LEIDEN_RESOLUTION_SHARED_GENES,
    UNS_NEIGHBORS_SHARED_GENES,
    UNS_SHARED_GENES,
)

logger = logging.getLogger(__name__)


def run_leiden_clustering_all_genes(
    adata_sc: AnnData,
    resolution: float,
    n_pca_comps: int = 30,
):
    """
    Leiden over-cluster the reference on all genes, in place; ``adata_sc.X`` is left raw.

    Requires: adata_sc.layers[LAYER_LOGNORM].
    Adds: adata_sc.obsm[OBSM_PCA], adata_sc.obs[OBS_LEIDEN_ALL_GENES],
    adata_sc.uns[UNS_LEIDEN_RESOLUTION_ALL_GENES],
    adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_ALL_GENES] (plus the scanpy neighbor graph).
    """

    n = min(n_pca_comps, adata_sc.n_obs - 1, adata_sc.n_vars - 1)
    sc.pp.pca(adata_sc, n_comps=n, layer=LAYER_LOGNORM, key_added=OBSM_PCA)

    sc.pp.neighbors(adata_sc, use_rep=OBSM_PCA)

    sc.tl.leiden(
        adata_sc,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        key_added=OBS_LEIDEN_ALL_GENES,
    )
    number_of_clusters = len(
        np.unique(adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).values)
    )
    logger.info(
        "Leiden clustering (resolution=%.2f): %d clusters",
        resolution,
        number_of_clusters,
    )

    adata_sc.uns[UNS_LEIDEN_RESOLUTION_ALL_GENES] = resolution
    adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_ALL_GENES] = number_of_clusters


def run_leiden_clustering_shared_genes(
    adata_sc: AnnData,
    resolution: float,
    n_pca_comps: int = 30,
):
    """
    Leiden over-cluster the reference restricted to the shared genes, in place on
    ``adata_sc``. PCA, neighbors, and Leiden run inside a throwaway shared-gene
    AnnData, then the results are copied back under the shared-gene keys.

    Requires: adata_sc.uns[UNS_SHARED_GENES], adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES].
    Adds: adata_sc.obsm[OBSM_PCA_SHARED_GENES], adata_sc.uns[UNS_NEIGHBORS_SHARED_GENES],
    adata_sc.obsp[OBSP_DISTANCES_SHARED_GENES], adata_sc.obsp[OBSP_CONNECTIVITIES_SHARED_GENES],
    adata_sc.obs[OBS_LEIDEN_SHARED_GENES], adata_sc.uns[UNS_LEIDEN_RESOLUTION_SHARED_GENES],
    adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_SHARED_GENES].
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

    sc.tl.leiden(
        adata_shared,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBS_LEIDEN_SHARED_GENES,
    )
    number_of_clusters = len(
        np.unique(adata_shared.obs[OBS_LEIDEN_SHARED_GENES].astype(int).values)
    )
    logger.info(
        "Leiden clustering on shared genes (resolution=%.2f): %d clusters",
        resolution,
        number_of_clusters,
    )

    # Copy the shared-gene PCA, neighbor graph, and cluster label back onto adata_sc.
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
    adata_sc.obs[OBS_LEIDEN_SHARED_GENES] = adata_shared.obs[
        OBS_LEIDEN_SHARED_GENES
    ].values
    adata_sc.uns[UNS_LEIDEN_RESOLUTION_SHARED_GENES] = resolution
    adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_SHARED_GENES] = number_of_clusters
