"""Disk outputs for the sweep: the per-cell start clustering (once per run) and,
per K, the spot->state mapping P (h5ad + CSV) and the start-cluster->state label map.

Also home to the shared reference-scaffold cache (``reference_scaffold.h5ad`` +
``reference_scaffold.meta.json``): the mapper-independent, prepared ``adata_sc``
computed once per sc/ST pair and reused by every mapper's sweep and the GUI.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from aim.adata_schema import (
    OBS_MAPPING_CONFIDENCE,
    OBS_START_CLUSTER,
    UNS_N_START_CLUSTERS,
    UNS_START_CLUSTER_NAMES,
)

logger = logging.getLogger(__name__)

# Bump whenever anything in the cached scaffold build changes (preprocessing,
# clustering params, aggregates, UMAP) or its stored keys are renamed, so stale
# caches self-invalidate. v2: leiden_* -> start_cluster_* keys.
REFERENCE_SCAFFOLD_FORMAT_VERSION = 2
_SCAFFOLD_H5AD = "reference_scaffold.h5ad"
_SCAFFOLD_META = "reference_scaffold.meta.json"

START_CLUSTERING_FILENAME = "start_clustering.h5ad"
START_CLUSTER_TO_STATE_FILENAME = "start_cluster_to_state.csv"


def write_start_clustering(output_folder: Path, adata_sc: anndata.AnnData) -> None:
    """
    Write start_clustering.h5ad (per-cell start cluster) to the run root.

    Two obs columns: ``start_cluster``, the integer label indexing
    ``start_cluster_to_state.csv``, and ``start_cluster_name``, its display name --
    the annotated cell type in start-from-annotation mode, ``cluster_<i>`` otherwise.

    Requires: adata_sc.obs[OBS_START_CLUSTER], adata_sc.uns[UNS_N_START_CLUSTERS],
    adata_sc.uns[UNS_START_CLUSTER_NAMES].
    """
    labels = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    names = [str(n) for n in adata_sc.uns[UNS_START_CLUSTER_NAMES]]
    if len(names) != int(adata_sc.uns[UNS_N_START_CLUSTERS]):
        raise ValueError(
            f"{UNS_START_CLUSTER_NAMES} has {len(names)} entries but "
            f"{UNS_N_START_CLUSTERS} is {adata_sc.uns[UNS_N_START_CLUSTERS]}"
        )
    anndata.AnnData(
        X=np.zeros((len(labels), 0), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "start_cluster": labels,
                # Categorical up front so anndata does not auto-convert on write
                # (which logs "storing 'start_cluster_name' as categorical").
                "start_cluster_name": pd.Categorical([names[i] for i in labels]),
            },
            index=adata_sc.obs_names,
        ),
    ).write_h5ad(output_folder / START_CLUSTERING_FILENAME)


def write_run_outputs(
    run_dir: Path,
    spot_to_state: np.ndarray,
    labels_k: np.ndarray,
    n_start_clusters: int,
    k: int,
    adata_st: anndata.AnnData,
    confidence: np.ndarray | None = None,
) -> None:
    """Write one K's outputs: start_cluster_to_state.csv and
    spot_to_state_mapping_soft.h5ad (P is S x K).

    ``confidence``, when the mapper defines it, is a (S,) array in [0, 1] stored
    as ``obs[OBS_MAPPING_CONFIDENCE]`` on the soft mapping h5ad; when ``None`` the
    column is simply omitted.
    """

    pd.DataFrame(
        {"start_cluster": np.arange(n_start_clusters), "state": labels_k}
    ).to_csv(run_dir / START_CLUSTER_TO_STATE_FILENAME, index=False)

    obs = pd.DataFrame(index=adata_st.obs_names)
    if confidence is not None:
        obs[OBS_MAPPING_CONFIDENCE] = np.asarray(confidence)

    anndata.AnnData(
        X=np.asarray(spot_to_state, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"state_{i}" for i in range(k)]),
    ).write_h5ad(run_dir / "spot_to_state_mapping_soft.h5ad")


# ---------------------------------------------------------------------------
# Shared reference-scaffold cache
# ---------------------------------------------------------------------------


def reference_scaffold_key(
    sc_path: Path,
    st_path: Path,
    shared_genes,
    leiden_resolution: float,
    start_from_annotation: str | None = None,
) -> dict:
    """Validity key for the cached reference scaffold.

    Depends on the sc file content (stat, not a content hash -- references are
    multi-GB), the shared-gene set (a sha256 of the sorted intersection, the true
    invariant the scaffold is built from), the Leiden resolution, and where the
    start clusters came from (``None`` = Leiden over-clustering, else the annotation
    column) -- otherwise a scaffold built one way would be reused for the other.
    ``st`` is keyed by stat too, but the shared-gene hash is what actually guards
    against an ST change that shifts the gene intersection.
    """
    sc_stat = Path(sc_path).stat()
    st_stat = Path(st_path).stat()
    genes = sorted(str(g) for g in shared_genes)
    gene_hash = hashlib.sha256("\x00".join(genes).encode("utf-8")).hexdigest()
    return {
        "format_version": REFERENCE_SCAFFOLD_FORMAT_VERSION,
        "sc_path": str(Path(sc_path).resolve()),
        "sc_size": sc_stat.st_size,
        "sc_mtime_ns": sc_stat.st_mtime_ns,
        "st_path": str(Path(st_path).resolve()),
        "st_size": st_stat.st_size,
        "st_mtime_ns": st_stat.st_mtime_ns,
        "shared_genes_sha256": gene_hash,
        "leiden_resolution": float(leiden_resolution),
        "start_from_annotation": start_from_annotation,
    }


def read_reference_scaffold(cache_dir: Path, key: dict) -> anndata.AnnData | None:
    """Return the cached scaffold ``adata_sc`` iff both files exist and the stored
    meta equals ``key``; any missing file, key mismatch, or read/parse error yields
    ``None`` (a cache miss), so a corrupt cache degrades to a recompute."""
    cache_dir = Path(cache_dir)
    h5ad_path = cache_dir / _SCAFFOLD_H5AD
    meta_path = cache_dir / _SCAFFOLD_META
    if not (h5ad_path.exists() and meta_path.exists()):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            stored = json.load(fh)
        if stored != key:
            return None
        adata_sc = anndata.read_h5ad(h5ad_path)
    except Exception:  # noqa: BLE001 - any corruption -> treat as a miss
        logger.warning("Ignoring unreadable reference scaffold cache in %s", cache_dir)
        return None
    logger.info("Reusing cached reference scaffold from %s", h5ad_path)
    return adata_sc


def write_reference_scaffold(
    cache_dir: Path, adata_sc: anndata.AnnData, key: dict
) -> None:
    """Persist the scaffold ``adata_sc`` + its validity key atomically.

    The h5ad is written first (temp file in the same dir + ``os.replace``), then
    the meta, so a present meta always implies a complete h5ad. Any failure (e.g.
    a Windows ``PermissionError``) is logged and swallowed -- the sweep continues
    without caching rather than aborting.
    """
    cache_dir = Path(cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".{os.getpid()}.tmp"
        h5ad_path = cache_dir / _SCAFFOLD_H5AD
        meta_path = cache_dir / _SCAFFOLD_META
        h5ad_tmp = h5ad_path.with_suffix(h5ad_path.suffix + suffix)
        meta_tmp = meta_path.with_suffix(meta_path.suffix + suffix)

        adata_sc.write_h5ad(h5ad_tmp)
        os.replace(h5ad_tmp, h5ad_path)

        with open(meta_tmp, "w", encoding="utf-8") as fh:
            json.dump(key, fh)
        os.replace(meta_tmp, meta_path)
        logger.info("Wrote reference scaffold cache to %s", h5ad_path)
    except Exception:  # noqa: BLE001 - caching is best-effort
        logger.warning(
            "Could not write reference scaffold cache to %s; continuing without cache.",
            cache_dir,
            exc_info=True,
        )
