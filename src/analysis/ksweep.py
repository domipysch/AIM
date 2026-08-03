"""Cross-K comparison: gather each K's key analysis metrics from its per-K
``analysis/data`` outputs into one table (``k_comparison.csv``) at the run root.

Reads the files each K's post-mapping analysis already writes (``cossim_summary.csv``,
``topology_metrics.json``, ``modularity_metrics.json``), so it runs after the K-loop
with no recomputation. Metrics missing for a given K come through as NaN.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    """Load a JSON dict, or return an empty dict if the file is absent/unreadable."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _finite(x) -> float:
    """Coerce a value to float, mapping non-finite (inf / NaN / None) to NaN so it
    leaves a gap in the plot rather than distorting an autoscaled axis."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def collect_k_metrics(output_folder: Path, ks: list[int]) -> pd.DataFrame:
    """Gather the per-K comparison metrics into one DataFrame (one row per K, ascending).

    Columns: ``k``, ``cossim_hard_norm_spot``, ``cossim_hard_norm_gene``,
    ``cossim_hard_raw_spot``, ``cossim_hard_raw_gene``, ``nhood_mean_self_zscore``,
    ``local_purity_zscore``, ``modularity_shared``, ``mean_confidence``.
    """
    rows = []
    for k in ks:
        data_dir = output_folder / f"k_{k:03d}" / "analysis" / "data"

        cossim_norm_spot = cossim_norm_gene = np.nan
        cossim_raw_spot = cossim_raw_gene = np.nan
        cossim_path = data_dir / "cossim_summary.csv"
        if cossim_path.exists():
            cs = pd.read_csv(cossim_path, index_col=0)
            if "hard-norm" in cs.index:
                cossim_norm_spot = float(cs.loc["hard-norm", "median_spot"])
                cossim_norm_gene = float(cs.loc["hard-norm", "median_gene"])
            if "hard-raw" in cs.index:
                cossim_raw_spot = float(cs.loc["hard-raw", "median_spot"])
                cossim_raw_gene = float(cs.loc["hard-raw", "median_gene"])

        topo = _load_json(data_dir / "topology_metrics.json")
        # z-scores can be non-finite (inf) when a permutation null has zero variance
        # for a degenerate/tiny state; coerce to NaN so they leave a gap instead of
        # blowing up the plot's autoscaled y-axis.
        # These sub-dicts are stored as null when a K has too few states for the
        # test (e.g. K=1), so coalesce None -> {} before indexing.
        nhood = topo.get("nhood_enrichment") or {}
        purity = topo.get("local_purity") or {}
        self_z = _finite(nhood.get("mean_self_zscore", np.nan))
        purity_z = _finite(purity.get("z_score", np.nan))

        modularity = _load_json(data_dir / "modularity_metrics.json")
        mod_shared = modularity.get("modularity_shared", np.nan)

        # Present only for mappers that define a confidence (nearest / wann);
        # absent -> NaN -> the curve simply does not appear for this run.
        confidence = _load_json(data_dir / "confidence_summary.json")
        mean_conf = confidence.get("mean", np.nan)

        rows.append(
            {
                "k": k,
                "cossim_hard_norm_spot": cossim_norm_spot,
                "cossim_hard_norm_gene": cossim_norm_gene,
                "cossim_hard_raw_spot": cossim_raw_spot,
                "cossim_hard_raw_gene": cossim_raw_gene,
                "nhood_mean_self_zscore": self_z,
                "local_purity_zscore": purity_z,
                "modularity_shared": mod_shared,
                "mean_confidence": mean_conf,
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def compare_k_runs(output_folder: Path, ks: list[int]) -> None:
    """Write ``k_comparison.csv`` at the run root: the swept K-runs' reconstruction
    cosine similarity, spatial organisation, and mapping modularity gathered into
    one table (the GUI renders the comparison plots from it on demand)."""
    output_folder = Path(output_folder)
    df = collect_k_metrics(output_folder, ks)
    csv_path = output_folder / "k_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("K-sweep comparison written to %s", csv_path)
