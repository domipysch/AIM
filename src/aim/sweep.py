"""The K sweep: over-cluster once, build the tree once, then for every K cut the tree,
assemble state profiles, map spots onto states, write outputs, and run the analysis."""

import logging
from pathlib import Path

import anndata
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
from analysis.ksweep import compare_k_runs
from .aggregation import compute_leiden_aggregates
from .clustering import (
    run_leiden_clustering_all_genes,
    run_leiden_clustering_shared_genes,
)
from .io import (
    read_reference_scaffold,
    reference_scaffold_key,
    write_leiden_overclustering_all_genes,
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
    adata_sc: anndata.AnnData, leiden_resolution: float
) -> None:
    """The mapper-independent reference build, in place on a pre-processed
    ``adata_sc``: over-cluster on all genes and on shared genes, aggregate per
    cluster, and compute both UMAP embeddings.

    This is the expensive part of a sweep (PCA + neighbors + Leiden x2 + UMAP x2)
    and it depends only on the reference, the shared-gene set, and the resolution
    -- so it is cached once per sc/ST pair and reused across mappers. Requires
    :func:`_preprocess_sc` to have run.
    """
    logger.info("Computing Leiden over-clustering...")
    run_leiden_clustering_all_genes(adata_sc, resolution=leiden_resolution)
    run_leiden_clustering_shared_genes(adata_sc, resolution=leiden_resolution)

    # Cluster labels come from the all-genes Leiden; expression from shared genes.
    compute_leiden_aggregates(adata_sc)

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
    output_folder: Path,
    mapper: SpotStateMapper,
    generate_pdf: bool,
    agglo_tree_method: str = "average",
    agglo_tree_metric: str = "cosine",
    leiden_resolution: float = 3.0,
    k_min: int | None = None,
    k_max: int | None = None,
    k_step: int = 1,
    reference_cache_dir: Path | None = None,
):
    """Run the full AIM sweep for one sc/ST pair, writing one folder per K under
    ``output_folder`` and running the post-mapping analysis on each.

    When ``reference_cache_dir`` is given, the mapper-independent reference
    scaffold (over-clustering + aggregates + UMAPs) is read from / written to a
    shared cache there, so running several mappers for one pair computes it once.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    adata_sc = anndata.read_h5ad(sc_path)
    adata_st = anndata.read_h5ad(st_path)

    shared_genes = list(adata_sc.var_names.intersection(adata_st.var_names))
    cached = None
    if reference_cache_dir is not None:
        key = reference_scaffold_key(sc_path, st_path, shared_genes, leiden_resolution)
        cached = read_reference_scaffold(reference_cache_dir, key)

    if cached is not None:
        # Reuse the shared scaffold; only the ST half of preprocessing is left.
        adata_sc = cached
        _preprocess_st(adata_st, adata_sc.uns[UNS_SHARED_GENES])
    else:
        pre_processing(adata_sc, adata_st)
        build_reference_scaffold(adata_sc, leiden_resolution)
        if reference_cache_dir is not None:
            write_reference_scaffold(reference_cache_dir, adata_sc, key)

    # Every mapper's run root keeps its own copy (the GUI reads it per mapper).
    write_leiden_overclustering_all_genes(output_folder, adata_sc)

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

    # Each K's subcluster->state cut, computed once and reused below. Also handed
    # to the mapper's one-time prepare() hook (the reference mapper materialises
    # its per-K state-labelled inputs from it).
    labels_by_k = {
        k: labels_at_k(agglomerative_clustering, k, n_leiden_clusters) for k in ks
    }
    mapper.prepare(adata_sc, adata_st, labels_by_k)

    for k in ks:

        leiden_to_state = labels_by_k[k]
        spot_to_state, confidence = mapper.map(leiden_to_state, k)

        run_dir = output_folder / f"k_{k:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        write_run_outputs(
            run_dir=run_dir,
            spot_to_state=spot_to_state,
            confidence=confidence,
            labels_k=leiden_to_state,
            n_leiden=n_leiden_clusters,
            k=k,
            adata_st=adata_st,
        )
        logger.info("K=%3d mapped -> %s", k, run_dir)

        run_analysis(adata_sc, adata_st, run_dir, generate_pdf)

    # Cross-K comparison: gather the per-K analysis metrics written above into one
    # table + figure at the run root (independent of the per-K PDF reports).
    logger.info("Comparing K-runs...")
    compare_k_runs(output_folder, ks)
