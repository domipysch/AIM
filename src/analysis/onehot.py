import json
import logging
from pathlib import Path
import pandas as pd
from anndata import AnnData

from adata_schema import (
    OBSM_MAPPING_SOFT,
)
from metrics.onehot import onehot_metrics
from plots import plot_dominance_thresholds, plot_onehot_distribution

logger = logging.getLogger(__name__)


def analyse_spot_to_state_one_hotness(
    adata_st: AnnData,
    data_dir: Path,
):
    """One-hotness metrics for the spot->state mapping P.

    Requires: adata_st.obsm[OBSM_MAPPING_SOFT].
    Writes onehot_per_row_mapping.csv and onehot_summary_mapping.json under
    data_dir, plus the distribution and threshold plots under plots_dir.
    """

    m = onehot_metrics(adata_st.obsm[OBSM_MAPPING_SOFT])

    pd.DataFrame(
        {
            "id": adata_st.obs_names,
            "max_prob": m["max_prob"],
            "gini_impurity": m["gini_impurity"],
            "entropy": m["entropy"],
        }
    ).to_csv(data_dir / "onehot_per_row_mapping.csv", index=False)
    with open(data_dir / "onehot_summary_mapping.json", "w") as f:
        json.dump(
            {"n_rows": m["n_rows"], "n_cols": m["n_cols"], "summary": m["summary"]},
            f,
            indent=2,
        )


def plot_spot_to_state_one_hotness(
    plots_dir: Path,
    data_dir: Path,
):
    """Render the spot->state one-hotness figures from the metrics on disk.

    Reads onehot_per_row_mapping.csv and onehot_summary_mapping.json (written by
    analyse_spot_to_state_one_hotness) from data_dir, and writes the distribution
    and threshold plots under plots_dir.
    """

    per_row = pd.read_csv(data_dir / "onehot_per_row_mapping.csv")
    with open(data_dir / "onehot_summary_mapping.json") as f:
        summary = json.load(f)

    m = {
        "max_prob": per_row["max_prob"].to_numpy(),
        "gini_impurity": per_row["gini_impurity"].to_numpy(),
        "entropy": per_row["entropy"].to_numpy(),
        "n_rows": summary["n_rows"],
        "n_cols": summary["n_cols"],
        "summary": summary["summary"],
    }

    plot_onehot_distribution(
        m, plots_dir / "onehot_distribution_mapping.png", row_label="spot"
    )
    plot_dominance_thresholds(
        m, plots_dir / "onehot_thresholds_mapping.png", row_label="spot"
    )
