"""Per-spot mapping-confidence distribution for the analysis report.

Only produced when the mapper defined a confidence (``obs[OBS_MAPPING_CONFIDENCE]``,
loaded by ``loading.py``); the learned and reference mappers define none, so the
step is a no-op for them and the report simply omits the section.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from adata_schema import OBS_MAPPING_CONFIDENCE
from plots import plot_confidence_distribution

logger = logging.getLogger(__name__)


def analyse_spot_confidence(adata_st: AnnData, data_dir: Path) -> None:
    """Summarise the per-spot mapping confidence, when the mapper defined one.

    Requires: adata_st.obs[OBS_MAPPING_CONFIDENCE] (skipped if absent). Writes
    confidence_per_spot.csv and confidence_summary.json under data_dir.
    """
    if OBS_MAPPING_CONFIDENCE not in adata_st.obs:
        logger.info("No per-spot confidence for this mapping; skipping.")
        return

    conf = adata_st.obs[OBS_MAPPING_CONFIDENCE].to_numpy(dtype=float)

    pd.DataFrame({"id": adata_st.obs_names, "confidence": conf}).to_csv(
        data_dir / "confidence_per_spot.csv", index=False
    )
    summary = {
        "n_spots": int(conf.size),
        "mean": float(np.mean(conf)),
        "median": float(np.median(conf)),
        "min": float(np.min(conf)),
        "max": float(np.max(conf)),
        "std": float(np.std(conf)),
        "frac_above_0.5": float(np.mean(conf >= 0.5)),
        "frac_above_0.9": float(np.mean(conf >= 0.9)),
    }
    with open(data_dir / "confidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def plot_spot_confidence(plots_dir: Path, data_dir: Path) -> None:
    """Render the confidence histogram from the metrics on disk, if present.

    Reads confidence_per_spot.csv and confidence_summary.json (written by
    analyse_spot_confidence) from data_dir; a no-op when they are absent (the
    mapper defined no confidence).
    """
    per_spot = data_dir / "confidence_per_spot.csv"
    summary_path = data_dir / "confidence_summary.json"
    if not per_spot.exists() or not summary_path.exists():
        return

    conf = pd.read_csv(per_spot)["confidence"].to_numpy()
    with open(summary_path) as f:
        summary = json.load(f)

    plot_confidence_distribution(
        conf, summary, plots_dir / "confidence_distribution.png"
    )
