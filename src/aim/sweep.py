"""The K sweep: over-cluster once, build the tree once, then for every K cut the tree,
assemble state profiles, map spots onto states, write outputs, and run the analysis."""

import logging
from pathlib import Path

import anndata
import torch
import scanpy as sc
from adata_schema import (
    UNS_LEIDEN_CENTROIDS_SHARED_GENES,
    UNS_SHARED_GENES,
    LAYER_LOGNORM,
    OBSM_LOGNORM_SHARED_GENES,
    UNS_LEIDEN_NUMBER_STATES_ALL_GENES,
    OBSM_UMAP,
    UNS_NEIGHBORS_SHARED_GENES,
    OBSM_UMAP_SHARED_GENES,
)
from analysis.analysis import run_analysis
from analysis.utils import to_dense
from .aggregation import (
    assemble_state_profiles_shared_genes,
    compute_leiden_aggregates,
)
from .clustering import (
    run_leiden_clustering_all_genes,
    run_leiden_clustering_shared_genes,
)
from .io import write_run_outputs, write_leiden_overclustering_all_genes
from .mapping import SpotStateMapper
from .tree import build_agglomeration_tree, labels_at_k

logger = logging.getLogger(__name__)


def pre_processing(
    adata_sc: anndata.AnnData,
    adata_st: anndata.AnnData,
):
    """
    Normalize and log1p the inputs into separate layers, keeping ``.X`` raw.

    Adds: adata_sc.uns[UNS_SHARED_GENES], adata_sc.layers[LAYER_LOGNORM],
    adata_st.layers[LAYER_LOGNORM], adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES],
    adata_st.obsm[OBSM_LOGNORM_SHARED_GENES]. The shared-gene variants recompute
    the size factor from shared-gene counts alone, so they cannot live in .layers.
    """

    shared_genes = list(adata_sc.var_names.intersection(adata_st.var_names))
    adata_sc.uns[UNS_SHARED_GENES] = shared_genes

    adata_sc.layers[LAYER_LOGNORM] = adata_sc.X.copy()
    sc.pp.normalize_total(adata_sc, target_sum=1e4, layer=LAYER_LOGNORM)
    sc.pp.log1p(adata_sc, layer=LAYER_LOGNORM)

    adata_st.layers[LAYER_LOGNORM] = adata_st.X.copy()
    sc.pp.normalize_total(adata_st, target_sum=1e4, layer=LAYER_LOGNORM)
    sc.pp.log1p(adata_st, layer=LAYER_LOGNORM)

    adata_sc_shared = adata_sc[:, shared_genes].copy()
    adata_sc_shared.uns.pop("log1p", None)  # prevent scanpy warning
    sc.pp.normalize_total(adata_sc_shared, target_sum=1e4)
    sc.pp.log1p(adata_sc_shared)
    adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES] = adata_sc_shared.X

    adata_st_shared = adata_st[:, shared_genes].copy()
    adata_st_shared.uns.pop("log1p", None)
    sc.pp.normalize_total(adata_st_shared, target_sum=1e4)
    sc.pp.log1p(adata_st_shared)
    adata_st.obsm[OBSM_LOGNORM_SHARED_GENES] = adata_st_shared.X


def run(
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
    mapper: SpotStateMapper,
    generate_pdf: bool,
    agglo_tree_method: str = "average",
    agglo_tree_metric: str = "sqeuclidean",
    leiden_resolution: float = 3.0,
    k_min: int | None = None,
    k_max: int | None = None,
    k_step: int = 1,
):
    """Run the full AIM sweep for one sc/ST pair, writing one folder per K under
    ``output_folder`` and running the post-mapping analysis on each."""

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    adata_sc = anndata.read_h5ad(sc_path)
    adata_st = anndata.read_h5ad(st_path)
    pre_processing(adata_sc, adata_st)

    logger.info("Computing Leiden over-clustering...")
    run_leiden_clustering_all_genes(adata_sc, resolution=leiden_resolution)
    run_leiden_clustering_shared_genes(adata_sc, resolution=leiden_resolution)

    write_leiden_overclustering_all_genes(output_folder, adata_sc)

    # Cluster labels come from the all-genes Leiden; expression from shared genes.
    compute_leiden_aggregates(adata_sc)

    # UMAP is used only by the analysis, not by the mapping.
    sc.tl.umap(adata_sc, key_added=OBSM_UMAP)
    sc.tl.umap(
        adata_sc,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBSM_UMAP_SHARED_GENES,
    )

    agglomerative_clustering = build_agglomeration_tree(
        adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES],
        method=agglo_tree_method,
        metric=agglo_tree_metric,
    )

    n_leiden_clusters = adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_ALL_GENES]
    k_hi = min(n_leiden_clusters, k_max) if k_max else n_leiden_clusters
    k_lo = max(1, k_min) if k_min else 1
    ks = list(range(k_hi, k_lo - 1, -k_step))
    logger.info(
        "Sweeping K from %d down to %d (step %d): %d levels",
        k_hi,
        k_lo,
        k_step,
        len(ks),
    )

    for k in ks:

        labels_k = labels_at_k(agglomerative_clustering, k, n_leiden_clusters)
        m_shared = assemble_state_profiles_shared_genes(labels_k, k, adata_sc)

        # Densify: adata_st.X is often sparse and torch.tensor can't consume it.
        z_shared = to_dense(adata_st[:, adata_sc.uns[UNS_SHARED_GENES]])
        spot_to_state = mapper.map(
            torch.tensor(z_shared, dtype=torch.float32),
            m_shared,
        )

        run_dir = output_folder / f"k_{k:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        write_run_outputs(
            run_dir=run_dir,
            spot_to_state=spot_to_state,
            labels_k=labels_k,
            n_leiden=n_leiden_clusters,
            k=k,
            adata_st=adata_st,
        )
        logger.info("K=%3d mapped -> %s", k, run_dir)

        run_analysis(adata_sc, adata_st, run_dir, generate_pdf)
