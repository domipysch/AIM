import json
import logging
from pathlib import Path
import pandas as pd
from anndata import AnnData

from adata_schema import (
    OBSM_MAPPING_SOFT,
)
from metrics.onehot import onehot_metrics
from metrics.onehot_plots import plot_dominance_thresholds, plot_onehot_distribution

logger = logging.getLogger(__name__)


def save_onehot(
    adata_st: AnnData,
    plots_dir: Path,
    data_dir: Path,
) -> dict:
    """Compute one-hotness metrics for one soft assignment matrix (e.g. P).

    Always returns the onehot_metrics dict so its summary can be folded into the
    flat objective_metrics.csv. When write_artifacts is True, also writes the
    per-row CSV / summary JSON and the distribution + threshold plots; pass False
    when only the returned metrics are needed (e.g. mapping_prob, whose report
    section was removed) to skip the now-unused on-disk outputs."""

    m = onehot_metrics(adata_st.obsm[OBSM_MAPPING_SOFT])
    pd.DataFrame(
        {
            "id": adata_st.obs_names,
            "max_prob": m["max_prob"],
            "gini_impurity": m["gini_impurity"],
            "entropy": m["entropy"],
        }
    ).to_csv(data_dir / f"onehot_per_row_mapping.csv", index=False)
    with open(data_dir / f"onehot_summary_mapping.json", "w") as f:
        json.dump(
            {"n_rows": m["n_rows"], "n_cols": m["n_cols"], "summary": m["summary"]},
            f,
            indent=2,
        )
    plot_onehot_distribution(
        m, plots_dir / f"onehot_distribution_mapping.png", row_label="spot"
    )
    plot_dominance_thresholds(
        m, plots_dir / f"onehot_thresholds_mapping.png", row_label="spot"
    )
    return m
