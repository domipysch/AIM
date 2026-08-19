"""Pure on-disk readers for AIM sweep outputs.

Everything here is cheap and side-effect-free (reads only); the Streamlit app
wraps the expensive ones with ``st.cache_data``. A "run root" is the folder a
single mapper's sweep wrote into (``<output_dir>/<mapper>/``): it holds
``config.yaml``, ``start_clustering.h5ad``, ``k_comparison.{csv,png}`` and
one ``k_<kkk>/`` folder per swept K.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# MAPPING_CHOICES is the authoritative list of valid mapper names.
from aim import MAPPING_CHOICES
from aim.io import START_CLUSTERING_FILENAME

_K_DIR_RE = re.compile(r"^k_(\d{3})$")


def run_root(output_dir: Path, mapper: str) -> Path:
    """The run-root folder for one mapper under the shared output dir."""
    return Path(output_dir) / mapper


def is_run_root(path: Path) -> bool:
    """A folder counts as a finished/started run root if it has a config or any K folder."""
    path = Path(path)
    if (path / "config.yaml").exists():
        return True
    return any(_K_DIR_RE.match(p.name) for p in path.glob("k_*") if p.is_dir())


def list_mappers(output_dir: Path) -> list[str]:
    """Mapper names (in MAPPING_CHOICES order) that already have a run root on disk."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    present = {p.name for p in output_dir.iterdir() if p.is_dir() and is_run_root(p)}
    return [m for m in MAPPING_CHOICES if m in present]


def k_dir(root: Path, k: int) -> Path:
    return Path(root) / f"k_{k:03d}"


def list_ks(root: Path) -> list[int]:
    """Sorted (ascending) list of K values that have a folder under ``root``."""
    root = Path(root)
    if not root.exists():
        return []
    ks: list[int] = []
    for p in root.glob("k_*"):
        m = _K_DIR_RE.match(p.name)
        if p.is_dir() and m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def data_dir(root: Path, k: int) -> Path:
    return k_dir(root, k) / "analysis" / "data"


def load_soft(root: Path, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(P, hard, confidence)`` for one K.

    ``P`` is the S x K soft matrix, ``hard`` its per-spot argmax, and
    ``confidence`` the per-spot value in [0, 1] or ``None`` when the mapper wrote
    none (tangram / tacco / dot).
    """
    path = k_dir(root, k) / "spot_to_state_mapping_soft.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path)
    P = np.asarray(adata.X, dtype=np.float64)
    hard = P.argmax(axis=1)
    confidence = None
    if "mapping_confidence" in adata.obs:
        confidence = adata.obs["mapping_confidence"].to_numpy(dtype=np.float64)
    return P, hard, confidence


def load_data_json(root: Path, k: int, name: str) -> dict | None:
    path = data_dir(root, k) / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_cossim_distributions(root: Path, k: int) -> dict[str, dict[str, list]]:
    """Per-combo reconstruction cosine-similarity value lists for one K.

    Returns ``{label: {"per_gene": [...], "per_spot": [...]}}`` for the four
    soft/hard x raw/norm combos found under ``analysis/data/cossim`` (empty dict
    if none). Each on-disk JSON stores ``{"values": {name: cossim}}``.
    """
    cdir = data_dir(root, k) / "cossim"
    out: dict[str, dict[str, list]] = {}
    for label in ("soft-raw", "hard-raw", "soft-norm", "hard-norm"):
        gene_path = cdir / f"cossim-per-gene-{label}.json"
        spot_path = cdir / f"cossim-per-spot-{label}.json"
        if not (gene_path.exists() and spot_path.exists()):
            continue
        with open(gene_path) as f:
            per_gene = list(json.load(f)["values"].values())
        with open(spot_path) as f:
            per_spot = list(json.load(f)["values"].values())
        out[label] = {"per_gene": per_gene, "per_spot": per_spot}
    return out


def ksweep_csv(root: Path) -> pd.DataFrame | None:
    path = Path(root) / "k_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


# Columns the sweep CSV gained after some runs were already computed. When a
# k_comparison.csv predates them they are recovered from each K's analysis outputs
# (same values, just not yet collected into the table), so an older run root still
# shows every curve without re-running the sweep. Nothing is computed here. The
# modularities carry no null — a shuffled modularity is meaningless, so the sweep
# never measures one and the coherence criterion simply has no crosshair.
_MODULARITY_COLUMNS = ("modularity_st_expression",)
# csv column -> cossim_summary.csv column, both on the "hard-raw" row
_COSSIM_NULL_COLUMNS = {
    "cossim_hard_raw_spot_null": "median_spot_null",
    "cossim_hard_raw_gene_null": "median_gene_null",
}


def _as_float(value) -> float:
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _cossim_summary_row(root: Path, k: int, row: str) -> dict:
    """One row of a K's ``cossim_summary.csv`` as a plain dict (empty if absent)."""
    path = data_dir(root, k) / "cossim_summary.csv"
    if not path.exists():
        return {}
    try:
        table = pd.read_csv(path, index_col=0)
    except (OSError, pd.errors.ParserError):
        return {}
    if row not in table.index:
        return {}
    return table.loc[row].to_dict()


def ksweep_table(root: Path) -> pd.DataFrame | None:
    """``k_comparison.csv``, back-filled for run roots written by an older sweep.

    The sweep collects every metric the GUI plots — including the label-shuffle
    nulls measured by the post-mapping analysis — so this is a plain read; only
    columns absent from the CSV are looked up in the per-K analysis outputs.
    Returns ``None`` if the CSV is absent.
    """
    df = ksweep_csv(root)
    if df is None or "k" not in df.columns:
        return df
    df = df.copy()
    ks = df["k"].astype(int).tolist()

    missing = [c for c in _MODULARITY_COLUMNS if c not in df.columns]
    if missing:
        per_k = [load_data_json(root, k, "modularity_metrics.json") or {} for k in ks]
        for column in missing:
            df[column] = [_as_float(m.get(column)) for m in per_k]

    missing = [c for c in _COSSIM_NULL_COLUMNS if c not in df.columns]
    if missing:
        per_k = [_cossim_summary_row(root, k, "hard-raw") for k in ks]
        for column in missing:
            source = _COSSIM_NULL_COLUMNS[column]
            df[column] = [_as_float(row.get(source)) for row in per_k]

    return df


def list_obs_columns(sc_path: Path) -> list[str]:
    """Column names of an h5ad's ``obs``, read in backed mode -- cheap even for a
    multi-GB reference. Used to offer annotation columns as start clusters."""
    adata = ad.read_h5ad(sc_path, backed="r")
    try:
        return [str(c) for c in adata.obs.columns]
    finally:
        if adata.isbacked:
            adata.file.close()


def load_spatial_coords(st_path: Path) -> np.ndarray | None:
    """Spatial coordinates (n_spots x 2) from the ST h5ad, or None if absent."""
    adata = ad.read_h5ad(st_path, backed="r")
    try:
        if "spatial" not in adata.obsm:
            return None
        return np.asarray(adata.obsm["spatial"])
    finally:
        if adata.isbacked:
            adata.file.close()


def start_clustering_path(root: Path) -> Path | None:
    """The run root's start-clustering h5ad, or ``None`` if it has neither the
    current file nor the pre-rename ``leiden_overclustering.h5ad``."""
    root = Path(root)
    for name in (START_CLUSTERING_FILENAME, "leiden_overclustering.h5ad"):
        path = root / name
        if path.exists():
            return path
    return None


def load_start_cluster_labels(root: Path) -> np.ndarray | None:
    """Per-cell integer start-cluster labels from start_clustering.h5ad at the run
    root, matching the ``start_cluster`` index in ``start_cluster_to_state.csv``.

    An older run root stores them as the category strings ``leiden_<i>`` in a
    ``leiden_cluster`` column; that layout is parsed back to ``i`` so existing
    results keep loading.
    """
    path = start_clustering_path(root)
    if path is None:
        return None
    obs = ad.read_h5ad(path).obs
    if "start_cluster" in obs:
        return obs["start_cluster"].to_numpy(dtype=int)
    names = obs["leiden_cluster"].astype(str).to_numpy()
    return np.array([int(n.split("_")[1]) for n in names], dtype=int)


def load_start_cluster_names(root: Path) -> list[str] | None:
    """Display name per start cluster (index-aligned), or ``None`` when the run root
    predates them -- an older ``leiden_overclustering.h5ad`` carries no names."""
    path = start_clustering_path(root)
    if path is None:
        return None
    obs = ad.read_h5ad(path).obs
    if "start_cluster" not in obs or "start_cluster_name" not in obs:
        return None
    labels = obs["start_cluster"].to_numpy(dtype=int)
    names = obs["start_cluster_name"].astype(str).to_numpy()
    out = [""] * (int(labels.max()) + 1)
    for label, name in zip(labels, names):
        out[label] = name
    return out


def _config_value(output_dir: Path, field: str):
    """Best-effort read of ``field`` from any mapper's config.yaml under the output
    dir. Returns ``None`` if no config records it (configs are advisory only)."""
    import yaml

    for cfg_path in Path(output_dir).glob("*/config.yaml"):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and cfg.get(field) is not None:
                return cfg[field]
        except Exception:  # noqa: BLE001 - config is advisory only
            continue
    return None


def leiden_resolution_from_config(output_dir: Path) -> float | None:
    """Best-effort read of the leiden resolution any mapper's config.yaml recorded."""
    value = _config_value(output_dir, "leiden_resolution")
    return float(value) if value is not None else None


def start_from_annotation_from_config(output_dir: Path) -> str | None:
    """Best-effort read of the start-cluster annotation column any mapper's
    config.yaml recorded, so re-opening an output dir rebuilds its scaffold the way
    the sweep built it. ``None`` means Leiden over-clustering."""
    value = _config_value(output_dir, "start_from_annotation")
    return str(value) if value is not None else None


def linkage_method_from_config(output_dir: Path) -> str | None:
    """Best-effort read of the agglomeration linkage any mapper's config.yaml
    recorded. ``None`` when no config records it (an older run root). Falls back to
    ``agglo_tree_method``, the key's name before it was renamed."""
    value = _config_value(output_dir, "linkage_method")
    if value is None:
        value = _config_value(output_dir, "agglo_tree_method")
    return str(value) if value is not None else None
