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
    plots_dir: Path,
    data_dir: Path,
):
    """Compute one-hotness metrics for the spot->state mapping P and write its
    artifacts (per-row CSV, summary JSON, distribution + threshold plots).

    Reads P from ``adata_st.obsm[OBSM_MAPPING_SOFT]``"""

    # Compute one-hotness metrics for spot to state mapping
    m = onehot_metrics(adata_st.obsm[OBSM_MAPPING_SOFT])

    # Save to file
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

    # Generate plots
    plot_onehot_distribution(
        m, plots_dir / "onehot_distribution_mapping.png", row_label="spot"
    )
    plot_dominance_thresholds(
        m, plots_dir / "onehot_thresholds_mapping.png", row_label="spot"
    )
