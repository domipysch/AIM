"""Load (or build) the reference AnnData "scaffold" the UMAP / profile /
merge-map plots need.

The scaffold is the reference with lognorm layers, all- and shared-gene Leiden
labels, per-cluster aggregates, and both UMAP embeddings -- exactly what a sweep
computes once per run via ``aim.sweep.build_reference_scaffold``. It is mapper-
and K-independent (only the state cut/colour changes per K), so the sweep caches
it once per (sc, ST) pair at ``<output_dir>/reference_scaffold.h5ad``. This
module reads that same shared cache, and builds + writes it on a miss so a later
sweep reuses it.

To guarantee the all-genes Leiden labels line up with each K's
``leiden_to_state.csv`` cut, the labels are overwritten with the persisted
``leiden_overclustering.h5ad`` labels when those are available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad

from adata_schema import (
    OBS_LEIDEN_ALL_GENES,
    UNS_LEIDEN_NUMBER_STATES_ALL_GENES,
)
from aim import io as aim_io
from aim.sweep import build_reference_scaffold, pre_processing

from . import data_access

logger = logging.getLogger(__name__)


def _build_sc(sc_path: Path, st_path: Path, resolution: float) -> ad.AnnData:
    """Recompute the reference scaffold from scratch (the mapper-independent half
    of a sweep), identical to what ``aim.sweep`` caches."""
    adata_sc = ad.read_h5ad(sc_path)
    adata_st = ad.read_h5ad(st_path)

    pre_processing(adata_sc, adata_st)  # lognorm layers + shared genes
    build_reference_scaffold(adata_sc, resolution)
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
    """Return the reference scaffold, reusing the sweep's shared cache at
    ``<output_dir>/reference_scaffold.h5ad`` and building + writing it on a miss so
    a later sweep reuses it in turn."""
    output_dir = Path(output_dir)
    key = aim_io.reference_scaffold_key(
        sc_path, st_path, _shared_genes(sc_path, st_path), resolution
    )
    adata_sc = aim_io.read_reference_scaffold(output_dir, key)
    if adata_sc is None:
        logger.info("Building reference scaffold (one-time)...")
        adata_sc = _build_sc(sc_path, st_path, resolution)
        aim_io.write_reference_scaffold(output_dir, adata_sc, key)

    _align_to_persisted_labels(adata_sc, output_dir)
    return adata_sc
