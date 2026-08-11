"""Load (or build) the reference AnnData "scaffold" the UMAP / profile /
merge-map plots need.

The scaffold is the reference with lognorm layers, start clusters, the shared-gene
Leiden labels, per-start-cluster aggregates, and both UMAP embeddings -- exactly
what a sweep computes once per run via ``aim.sweep.build_reference_scaffold``. It is
mapper- and K-independent (only the state cut/colour changes per K), so the sweep
caches it once per (sc, ST) pair at ``<output_dir>/reference_scaffold.h5ad``. This
module reads that same shared cache, and builds + writes it on a miss so a later
sweep reuses it.

``start_from_annotation`` must match the sweep that wrote the output dir (see
``data_access.start_from_annotation_from_config``); it selects the start clusters
and is part of the cache key.

To guarantee the start-cluster labels line up with each K's
``start_cluster_to_state.csv`` cut, the labels are overwritten with the persisted
``start_clustering.h5ad`` labels when those are available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad

from aim.adata_schema import (
    OBS_START_CLUSTER,
    UNS_N_START_CLUSTERS,
    UNS_START_CLUSTER_NAMES,
)
from aim import io as aim_io
from aim.clustering import drop_unannotated
from aim.sweep import build_reference_scaffold, pre_processing

from . import data_access

logger = logging.getLogger(__name__)


def _build_sc(
    sc_path: Path,
    st_path: Path,
    resolution: float,
    start_from_annotation: str | None = None,
) -> ad.AnnData:
    """Recompute the reference scaffold from scratch (the mapper-independent half
    of a sweep), identical to what ``aim.sweep`` caches."""
    adata_sc = ad.read_h5ad(sc_path)
    adata_st = ad.read_h5ad(st_path)
    if start_from_annotation is not None:
        adata_sc = drop_unannotated(adata_sc, start_from_annotation)

    pre_processing(adata_sc, adata_st)  # lognorm layers + shared genes
    build_reference_scaffold(
        adata_sc, resolution, start_from_annotation=start_from_annotation
    )
    return adata_sc


def _shared_genes(sc_path: Path, st_path: Path) -> list[str]:
    """Shared-gene intersection, reading only var_names (backed mode) -- cheap."""
    sc_b = ad.read_h5ad(sc_path, backed="r")
    st_b = ad.read_h5ad(st_path, backed="r")
    try:
        return list(sc_b.var_names.intersection(st_b.var_names))
    finally:
        for a in (sc_b, st_b):
            if a.isbacked:
                a.file.close()


def _align_to_persisted_labels(adata_sc: ad.AnnData, output_dir: Path) -> None:
    """Overwrite the start-cluster labels with the sweep's persisted labels.

    Any run root's ``start_clustering.h5ad`` is authoritative and shared across
    mappers (same reference, same settings). Aligning guarantees the
    ``start_cluster_to_state.csv`` cut indexes the scaffold correctly regardless of
    any Leiden non-determinism.
    """
    for mapper in data_access.list_mappers(output_dir):
        root = data_access.run_root(output_dir, mapper)
        labels = data_access.load_start_cluster_labels(root)
        if labels is None:
            continue
        if len(labels) != adata_sc.n_obs:
            logger.warning(
                "Persisted start-cluster labels (%d) do not match scaffold cells "
                "(%d); keeping recomputed labels.",
                len(labels),
                adata_sc.n_obs,
            )
            return
        adata_sc.obs[OBS_START_CLUSTER] = labels.astype(str)
        adata_sc.uns[UNS_N_START_CLUSTERS] = int(labels.max()) + 1
        names = data_access.load_start_cluster_names(root)
        if names is not None:
            adata_sc.uns[UNS_START_CLUSTER_NAMES] = names
        logger.info("Aligned scaffold start clusters to the persisted labels.")
        return


def load_or_build_sc(
    sc_path: Path,
    st_path: Path,
    output_dir: Path,
    resolution: float,
    start_from_annotation: str | None = None,
) -> ad.AnnData:
    """Return the reference scaffold, reusing the sweep's shared cache at
    ``<output_dir>/reference_scaffold.h5ad`` and building + writing it on a miss so
    a later sweep reuses it in turn."""
    output_dir = Path(output_dir)
    key = aim_io.reference_scaffold_key(
        sc_path,
        st_path,
        _shared_genes(sc_path, st_path),
        resolution,
        start_from_annotation,
    )
    adata_sc = aim_io.read_reference_scaffold(output_dir, key)
    if adata_sc is None:
        logger.info("Building reference scaffold (one-time)...")
        adata_sc = _build_sc(sc_path, st_path, resolution, start_from_annotation)
        aim_io.write_reference_scaffold(output_dir, adata_sc, key)

    _align_to_persisted_labels(adata_sc, output_dir)
    return adata_sc
