"""Pure on-disk readers for AIM sweep outputs.

Everything here is cheap and side-effect-free (reads only); the Streamlit app
wraps the expensive ones with ``st.cache_data``. A "run root" is the folder a
single mapper's sweep wrote into (``<output_dir>/<mapper>/``): it holds
``config.yaml``, ``leiden_overclustering.h5ad``, ``k_comparison.{csv,png}`` and
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
    none (learned / tangram / tacco / dot).
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


def has_confidence(root: Path, k: int) -> bool:
    """Whether this K's soft output carries a per-spot confidence column."""
    path = k_dir(root, k) / "spot_to_state_mapping_soft.h5ad"
    if not path.exists():
        return False
    # Read obs only (cheap) via backed mode.
    adata = ad.read_h5ad(path, backed="r")
    try:
        return "mapping_confidence" in adata.obs
    finally:
        if adata.isbacked:
            adata.file.close()


def load_data_json(root: Path, k: int, name: str) -> dict | None:
    path = data_dir(root, k) / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_data_csv(root: Path, k: int, name: str, **kwargs) -> pd.DataFrame | None:
    path = data_dir(root, k) / name
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


def ksweep_png(root: Path) -> Path | None:
    path = Path(root) / "k_comparison.png"
    return path if path.exists() else None


def ksweep_csv(root: Path) -> pd.DataFrame | None:
    path = Path(root) / "k_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


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


def load_leiden_labels(root: Path) -> np.ndarray | None:
    """Per-cell integer Leiden labels from leiden_overclustering.h5ad at the run root.

    Values are stored as the category strings ``leiden_<i>``; this returns the
    integer ``i`` per cell, matching the ``leiden_cluster`` index in
    ``leiden_to_state.csv``.
    """
    path = Path(root) / "leiden_overclustering.h5ad"
    if not path.exists():
        return None
    adata = ad.read_h5ad(path)
    names = adata.obs["leiden_cluster"].astype(str).to_numpy()
    return np.array([int(n.split("_")[1]) for n in names], dtype=int)


def leiden_resolution_from_config(output_dir: Path) -> float | None:
    """Best-effort read of the leiden resolution any mapper's config.yaml recorded."""
    import yaml

    output_dir = Path(output_dir)
    for cfg_path in output_dir.glob("*/config.yaml"):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and "leiden_resolution" in cfg:
                return float(cfg["leiden_resolution"])
        except Exception:  # noqa: BLE001 - config is advisory only
            continue
    return None
