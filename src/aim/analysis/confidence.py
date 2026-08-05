"""Per-spot mapping-confidence summary for the post-mapping analysis.

Only produced when the mapper defined a confidence (``obs[OBS_MAPPING_CONFIDENCE]``,
loaded by ``loading.py``); the reference mappers define none, so the step is a
no-op for them and the GUI simply omits the confidence view.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from aim.adata_schema import OBS_MAPPING_CONFIDENCE

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
