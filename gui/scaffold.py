"""Build (and disk-cache) the reference AnnData "scaffold" the UMAP / profile /
merge-map plots need.

The scaffold is the reference with lognorm layers, all- and shared-gene Leiden
labels, both UMAP embeddings, and shared genes -- exactly what ``sweep.run``
computes once per run (``src/aim/sweep.py`` lines 91-108). It is mapper- and
K-independent (only the state cut/colour changes per K), so it is built once per
(sc, ST) pair and cached to ``<output_dir>/.gui_cache/scaffold_sc.h5ad``.

To guarantee the all-genes Leiden labels line up with each K's
``leiden_to_state.csv`` cut, the recomputed labels are overwritten with the
persisted ``leiden_overclustering.h5ad`` labels when those are available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from adata_schema import (
    OBS_LEIDEN_ALL_GENES,
    OBSM_UMAP,
    OBSM_UMAP_SHARED_GENES,
    UNS_LEIDEN_NUMBER_STATES_ALL_GENES,
    UNS_NEIGHBORS_SHARED_GENES,
)
from aim.clustering import (
    run_leiden_clustering_all_genes,
    run_leiden_clustering_shared_genes,
)
from aim.sweep import pre_processing

from . import data_access

logger = logging.getLogger(__name__)

_SCAFFOLD_FILE = "scaffold_sc.h5ad"


def _build_sc(sc_path: Path, st_path: Path, resolution: float) -> ad.AnnData:
    """Recompute the reference scaffold from scratch (the clustering half of a run)."""
    adata_sc = ad.read_h5ad(sc_path)
    adata_st = ad.read_h5ad(st_path)

    pre_processing(adata_sc, adata_st)  # lognorm layers + shared genes
    run_leiden_clustering_all_genes(adata_sc, resolution=resolution)
    run_leiden_clustering_shared_genes(adata_sc, resolution=resolution)

    # UMAP is used only for visualisation, exactly as in sweep.run.
    sc.tl.umap(adata_sc, key_added=OBSM_UMAP)
    sc.tl.umap(
        adata_sc,
        neighbors_key=UNS_NEIGHBORS_SHARED_GENES,
        key_added=OBSM_UMAP_SHARED_GENES,
    )
    return adata_sc


def _align_to_persisted_labels(adata_sc: ad.AnnData, output_dir: Path) -> None:
    """Overwrite the all-genes Leiden labels with the sweep's persisted labels.

    Any run root's ``leiden_overclustering.h5ad`` is authoritative and shared
    across mappers (same reference, same resolution). Aligning guarantees the
    ``leiden_to_state.csv`` cut indexes the scaffold correctly regardless of any
    Leiden non-determinism.
    """
    for mapper in data_access.list_mappers(output_dir):
        labels = data_access.load_leiden_labels(
            data_access.run_root(output_dir, mapper)
        )
        if labels is None:
            continue
        if len(labels) != adata_sc.n_obs:
            logger.warning(
                "Persisted Leiden labels (%d) do not match scaffold cells (%d); "
                "keeping recomputed labels.",
                len(labels),
                adata_sc.n_obs,
            )
            return
        adata_sc.obs[OBS_LEIDEN_ALL_GENES] = labels.astype(str)
        adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_ALL_GENES] = int(labels.max()) + 1
        logger.info("Aligned scaffold Leiden labels to persisted overclustering.")
        return


def load_or_build_sc(
    sc_path: Path,
    st_path: Path,
    output_dir: Path,
    resolution: float,
) -> ad.AnnData:
    """Return the reference scaffold, building + disk-caching it on first use."""
    cache_dir = Path(output_dir) / ".gui_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _SCAFFOLD_FILE

    if cache_path.exists():
        logger.info("Loading cached reference scaffold from %s", cache_path)
        adata_sc = ad.read_h5ad(cache_path)
    else:
        logger.info("Building reference scaffold (one-time)...")
        adata_sc = _build_sc(sc_path, st_path, resolution)
        adata_sc.write_h5ad(cache_path)

    _align_to_persisted_labels(adata_sc, output_dir)
    return adata_sc


def read_st(st_path: Path) -> ad.AnnData:
    """Fresh ST AnnData (raw X + spatial + obs_names) for per-K plot rendering."""
    return ad.read_h5ad(st_path)
