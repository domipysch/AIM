"""The K sweep: build the start clusters once, build the tree once, then for every K
cut the tree, assemble state profiles, map spots onto states, write outputs, and run
the analysis.

The start clusters -- the partition the agglomeration tree is built over -- are either
a Leiden over-clustering of the reference (the default) or a pre-existing annotation
(``start_from_annotation``); everything after that is identical.
"""

import logging
from pathlib import Path

import anndata
import scanpy as sc
from aim.adata_schema import (
    UNS_SHARED_GENES,
    LAYER_LOGNORM,
    OBSM_LOGNORM_SHARED_GENES,
    OBSM_UMAP,
    UNS_N_START_CLUSTERS,
    UNS_NEIGHBORS_SHARED_GENES,
    OBSM_UMAP_SHARED_GENES,
    UNS_START_CLUSTER_CENTROIDS_SHARED_GENES,
)
from aim.aim_config import LINKAGE_METHODS
from aim.analysis.analysis import run_analysis
from aim.analysis.ksweep import compare_k_runs
from .aggregation import compute_start_cluster_aggregates
from .clustering import (
    build_neighbor_graph_all_genes,
    drop_unannotated,
    run_leiden_clustering_all_genes,
    run_leiden_clustering_shared_genes,
    set_start_clusters_from_annotation,
)
from .io import (
    read_reference_scaffold,
    reference_scaffold_key,
    write_start_clustering,
    write_reference_scaffold,
    write_run_outputs,
)
from .mapping import SpotStateMapper
from .tree import build_agglomeration_tree, labels_at_k

logger = logging.getLogger(__name__)


def _lognorm_shared(adata: anndata.AnnData, shared_genes) -> None:
    """Add the shared-gene lognorm block to ``adata.obsm[OBSM_LOGNORM_SHARED_GENES]``.

    The size factor is recomputed from shared-gene counts alone, so it cannot live
    in ``.layers`` alongside the all-gene lognorm.
    """
    adata_shared = adata[:, list(shared_genes)].copy()
    adata_shared.uns.pop("log1p", None)  # prevent scanpy warning
    sc.pp.normalize_total(adata_shared, target_sum=1e4)
    sc.pp.log1p(adata_shared)
    adata.obsm[OBSM_LOGNORM_SHARED_GENES] = adata_shared.X


def _preprocess_sc(adata_sc: anndata.AnnData, shared_genes) -> None:
    """Reference half of :func:`pre_processing`: record the shared genes, add the
    all-gene lognorm layer, and add the shared-gene lognorm block."""
    adata_sc.uns[UNS_SHARED_GENES] = list(shared_genes)
    adata_sc.layers[LAYER_LOGNORM] = adata_sc.X.copy()
    sc.pp.normalize_total(adata_sc, target_sum=1e4, layer=LAYER_LOGNORM)
    sc.pp.log1p(adata_sc, layer=LAYER_LOGNORM)
    _lognorm_shared(adata_sc, shared_genes)


def _preprocess_st(adata_st: anndata.AnnData, shared_genes) -> None:
    """Spatial half of :func:`pre_processing`: add the all-gene lognorm layer and
    the shared-gene lognorm block. ``shared_genes`` may be a list or an ndarray
    (a cached ``uns[UNS_SHARED_GENES]`` reads back as an ndarray)."""
    adata_st.layers[LAYER_LOGNORM] = adata_st.X.copy()
    sc.pp.normalize_total(adata_st, target_sum=1e4, layer=LAYER_LOGNORM)
    sc.pp.log1p(adata_st, layer=LAYER_LOGNORM)
    _lognorm_shared(adata_st, shared_genes)


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
    _preprocess_sc(adata_sc, shared_genes)
    _preprocess_st(adata_st, shared_genes)


def build_reference_scaffold(
    adata_sc: anndata.AnnData,
    leiden_resolution: float,
    start_from_annotation: str | None = None,
) -> None:
    """The mapper-independent reference build, in place on a pre-processed
    ``adata_sc``: determine the start clusters, cluster on shared genes, aggregate
    per start cluster, and compute both UMAP embeddings.

    ``start_from_annotation`` selects where the start clusters come from: ``None``
    (the default) Leiden-over-clusters the reference on all genes; an ``obs`` column
    name instead takes that annotation as the starting partition, in which case no
    all-gene Leiden is computed -- only its neighbor graph, which the modularity
    metrics and the UMAP still need.

    This is the expensive part of a sweep (PCA + neighbors + Leiden + UMAP x2) and
    it depends only on the reference, the shared-gene set, the resolution and the
    start-cluster source -- so it is cached once per sc/ST pair and reused across
    mappers. Requires :func:`_preprocess_sc` to have run.
    """
    if start_from_annotation is None:
        logger.info("Computing Leiden over-clustering...")
        run_leiden_clustering_all_genes(adata_sc, resolution=leiden_resolution)
    else:
        # No over-clustering to compute: the annotated types are the start
        # clusters. The all-gene neighbor graph is still needed (modularity_all
        # measures against it, and the all-gene UMAP is laid out from it).
        logger.info("Start clusters from annotation %r...", start_from_annotation)
        build_neighbor_graph_all_genes(adata_sc)
        set_start_clusters_from_annotation(adata_sc, start_from_annotation)

    # A property of the reference, not a starting partition: always computed.
    run_leiden_clustering_shared_genes(adata_sc, resolution=leiden_resolution)

    # Cluster labels come from the start clusters; expression from shared genes.
    compute_start_cluster_aggregates(adata_sc)

    # UMAP is used only by the analysis, not by the mapping.
    sc.tl.umap(adata_sc, key_added=OBSM_UMAP)
    sc.tl.umap(
        adata_sc,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBSM_UMAP_SHARED_GENES,
    )


def run(
    sc_path: Path,
    st_path: Path,
    root_output_folder: Path,
    mapper: SpotStateMapper,
    linkage_method: str = LINKAGE_METHODS[0],
    leiden_resolution: float = 3.0,
    k_min: int | None = None,
    k_max: int | None = None,
    k_step: int = 1,
    start_from_annotation: str | None = None,
):
    """Run the full AIM sweep for one sc/ST pair, writing one folder per K under
    ``root_output_folder / mapper.name`` and running the post-mapping analysis on
    each.

    ``start_from_annotation`` is the ``obs`` column of a pre-existing cell-type
    annotation to use as the start clusters instead of a Leiden over-clustering; the
    sweep then runs K from the number of annotated types down. Cells with no label
    are dropped. Keep such runs in their own ``root_output_folder``: a run root is
    named after the mapper alone, so the two modes would otherwise overwrite each
    other.

    The mapper-independent reference scaffold (start clusters + aggregates + UMAPs)
    is cached under ``root_output_folder`` (see ``reference_scaffold_key`` /
    ``read_reference_scaffold``), so running several mappers for one pair computes
    it once.
    """

    root_output_folder = Path(root_output_folder)
    root_output_folder.mkdir(parents=True, exist_ok=True)
    mapping_output_folder = Path(root_output_folder) / mapper.name
    mapping_output_folder.mkdir(parents=True, exist_ok=True)

    adata_sc = anndata.read_h5ad(sc_path)
    adata_st = anndata.read_h5ad(st_path)

    if start_from_annotation is not None:
        # Before preprocessing, so every later block covers the same cells.
        adata_sc = drop_unannotated(adata_sc, start_from_annotation)

    shared_genes = list(adata_sc.var_names.intersection(adata_st.var_names))
    key = reference_scaffold_key(
        sc_path, st_path, shared_genes, leiden_resolution, start_from_annotation
    )
    cached = read_reference_scaffold(root_output_folder, key)

    if cached is not None:
        # Reuse the shared scaffold; only the ST half of preprocessing is left.
        adata_sc = cached
        _preprocess_st(adata_st, adata_sc.uns[UNS_SHARED_GENES])
    else:
        pre_processing(adata_sc, adata_st)
        build_reference_scaffold(
            adata_sc, leiden_resolution, start_from_annotation=start_from_annotation
        )
        if root_output_folder is not None:
            write_reference_scaffold(root_output_folder, adata_sc, key)

    # Every mapper's run root keeps its own copy (the GUI reads it per mapper).
    write_start_clustering(mapping_output_folder, adata_sc)

    agglomerative_clustering = build_agglomeration_tree(
        adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES],
        method=linkage_method,
    )

    n_start_clusters = adata_sc.uns[UNS_N_START_CLUSTERS]
    k_hi = min(n_start_clusters, k_max) if k_max else n_start_clusters
    k_lo = max(1, k_min) if k_min else 1
    ks = list(range(k_hi, k_lo - 1, -k_step))
    logger.info(
        "Sweeping K from %d down to %d (step %d): %d levels",
        k_hi,
        k_lo,
        k_step,
        len(ks),
    )

    # Each K's start-cluster->state cut, computed once and reused below. Also handed
    # to the mapper's one-time prepare() hook (the reference mapper materialises
    # its per-K state-labelled inputs from it).
    labels_by_k = {
        k: labels_at_k(agglomerative_clustering, k, n_start_clusters) for k in ks
    }
    mapper.prepare(adata_sc, adata_st, labels_by_k)

    try:
        for k in ks:

            start_cluster_to_state = labels_by_k[k]
            spot_to_state, confidence = mapper.map(start_cluster_to_state, k)

            run_dir = mapping_output_folder / f"k_{k:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            write_run_outputs(
                run_dir=run_dir,
                spot_to_state=spot_to_state,
                confidence=confidence,
                labels_k=start_cluster_to_state,
                n_start_clusters=n_start_clusters,
                k=k,
                adata_st=adata_st,
            )
            logger.info("K=%3d mapped -> %s", k, run_dir)

            run_analysis(adata_sc, adata_st, run_dir)
    finally:
        # Release any per-sweep resources (e.g. the reference aligner's worker
        # subprocess) whether the loop finished or raised.
        mapper.close()

    # Cross-K comparison: gather the per-K analysis metrics written above into one
    # table + figure at the run root (independent of the per-K PDF reports).
    logger.info("Comparing K-runs...")
    compare_k_runs(mapping_output_folder, ks)
