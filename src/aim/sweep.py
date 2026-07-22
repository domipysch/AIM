"""
The agglomerative-K sweep: over-cluster once, build the tree once, then for every
K cut the tree, assemble state profiles, map spots onto states, and run the full
post-mapping analysis for that K.

This module wires together the clustering half (``aggregation`` + ``tree``), the
modular mapping half (a ``SpotStateMapper``), the per-K disk I/O, and the
post-mapping analysis (``analysis.analysis.run_analysis``, called once per K
right after that K's outputs are written).
"""

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

    # Compute shared genes
    shared_genes = list(adata_sc.var_names.intersection(adata_st.var_names))
    adata_sc.uns[UNS_SHARED_GENES] = shared_genes

    # Compute log layers (Normalize + log1p into a separate layer so adata_sc.X keeps the raw counts)
    adata_sc.layers[LAYER_LOGNORM] = adata_sc.X.copy()
    sc.pp.normalize_total(adata_sc, target_sum=1e4, layer=LAYER_LOGNORM)
    sc.pp.log1p(adata_sc, layer=LAYER_LOGNORM)

    adata_st.layers[LAYER_LOGNORM] = adata_st.X.copy()
    sc.pp.normalize_total(adata_st, target_sum=1e4, layer=LAYER_LOGNORM)
    sc.pp.log1p(adata_st, layer=LAYER_LOGNORM)

    # Normalize + log1p restricted to the shared genes only (size factor computed
    # from shared-gene counts alone, not the full transcriptome/panel). G_shared
    # != n_vars, so this can't live in .layers -> stored in obsm instead, column-
    # aligned to shared_genes (== uns[UNS_SHARED_GENES]).
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
    leiden_resolution: float = 3.0,
    k_min: int | None = None,
    k_max: int | None = None,
    k_step: int = 1,
):

    # Write folder where to write output to
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load input single-cell and spatial data (raw, unnormalized)
    adata_sc = anndata.read_h5ad(sc_path)
    adata_st = anndata.read_h5ad(st_path)
    pre_processing(adata_sc, adata_st)

    # Compute Leiden over-clustering on single-cell data on all genes and only on shared genes
    logger.info("Computing Leiden over-clustering...")
    run_leiden_clustering_all_genes(adata_sc, resolution=leiden_resolution)
    run_leiden_clustering_shared_genes(adata_sc, resolution=leiden_resolution)

    # Save the leiden overclustering to file
    write_leiden_overclustering_all_genes(output_folder, adata_sc)

    # Aggregate the sc data over its Leiden clusters, raw and normalized (stored on adata_sc.uns).
    # Use Leiden clustering computed on all genes, but normalized values from only shared genes.
    compute_leiden_aggregates(adata_sc)

    # Compute UMAP coordinates considering all genes and shared genes only
    # (not necessary for mapping, but for analysis)
    sc.tl.umap(adata_sc, key_added=OBSM_UMAP)
    sc.tl.umap(
        adata_sc,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBSM_UMAP_SHARED_GENES,
    )

    # Build agglomerative clustering of Leiden clusters based on centroids only on shared genes
    agglomerative_clustering = build_agglomeration_tree(
        adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES]
    )

    # Get range of Ks to run the mapping on
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

    # Compute spot to state mapping for every k
    for k in ks:

        # Reconstruct cluster for given k ouf hierarchical cluster (L,)
        labels_k = labels_at_k(agglomerative_clustering, k, n_leiden_clusters)

        # Get cluster centroid expressions on shared genes (K, G_shared)
        m_shared = assemble_state_profiles_shared_genes(labels_k, k, adata_sc)

        # Compute mapping with given mapper (S x K)
        # Using shared genes raw count data
        spot_to_state = mapper.map(
            torch.tensor(
                adata_st[:, adata_sc.uns[UNS_SHARED_GENES]].X, dtype=torch.float32
            ),
            m_shared,
        )

        # Create folder for this k-run
        run_dir = output_folder / f"k_{k:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write outputs for this k-run to folder
        write_run_outputs(
            run_dir=run_dir,
            spot_to_state=spot_to_state,
            labels_k=labels_k,
            n_leiden=n_leiden_clusters,
            k=k,
            adata_st=adata_st,
        )
        logger.info("K=%3d mapped -> %s", k, run_dir)

        # Run analysis for this run
        run_analysis(adata_sc, adata_st, run_dir)
