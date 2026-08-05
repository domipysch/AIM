"""Disk outputs for the sweep: the per-cell Leiden over-clustering (once per run) and,
per K, the spot->state mapping P (h5ad + CSV) and the subcluster->state label map.

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
    OBS_LEIDEN_ALL_GENES,
    OBS_MAPPING_CONFIDENCE,
    UNS_LEIDEN_NUMBER_STATES_ALL_GENES,
)

logger = logging.getLogger(__name__)

# Bump whenever anything in the cached scaffold build changes (preprocessing,
# Leiden params, aggregates, UMAP), so stale caches self-invalidate.
REFERENCE_SCAFFOLD_FORMAT_VERSION = 1
_SCAFFOLD_H5AD = "reference_scaffold.h5ad"
_SCAFFOLD_META = "reference_scaffold.meta.json"


def write_leiden_overclustering_all_genes(
    output_folder: Path, adata_sc: anndata.AnnData
) -> None:
    """
    Write leiden_overclustering.h5ad (obs["leiden_cluster"] per cell) to the run root.

    Requires: adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_ALL_GENES], adata_sc.obs[OBS_LEIDEN_ALL_GENES].
    """
    leiden_names = [
        f"leiden_{i}" for i in range(adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_ALL_GENES])
    ]
    leiden_labels = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    cell_cluster_names = [leiden_names[i] for i in leiden_labels]
    anndata.AnnData(
        X=np.zeros((len(cell_cluster_names), 0), dtype=np.float32),
        obs=pd.DataFrame(
            # Store as categorical up front so anndata does not auto-convert it
            # on write (which logs "storing 'leiden_cluster' as categorical").
            {"leiden_cluster": pd.Categorical(cell_cluster_names)},
            index=adata_sc.obs_names,
        ),
    ).write_h5ad(output_folder / "leiden_overclustering.h5ad")


def write_run_outputs(
    run_dir: Path,
    spot_to_state: np.ndarray,
    labels_k: np.ndarray,
    n_leiden: int,
    k: int,
    adata_st: anndata.AnnData,
    confidence: np.ndarray | None = None,
) -> None:
    """Write one K's outputs: leiden_to_state.csv and spot_to_state_mapping_soft.h5ad (P is S x K).

    ``confidence``, when the mapper defines it, is a (S,) array in [0, 1] stored
    as ``obs[OBS_MAPPING_CONFIDENCE]`` on the soft mapping h5ad; when ``None`` the
    column is simply omitted.
    """

    pd.DataFrame({"leiden_cluster": np.arange(n_leiden), "state": labels_k}).to_csv(
        run_dir / "leiden_to_state.csv", index=False
    )

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
) -> dict:
    """Validity key for the cached reference scaffold.

    Depends on the sc file content (stat, not a content hash -- references are
    multi-GB), the shared-gene set (a sha256 of the sorted intersection, the true
    invariant the scaffold is built from), and the Leiden resolution. ``st`` is
    keyed by stat too, but the shared-gene hash is what actually guards against an
    ST change that shifts the gene intersection.
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
