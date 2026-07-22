"""Generic one-hotness metrics for any soft row-stochastic assignment matrix.

Works on any (n_rows x n_cols) matrix whose rows are probability-like vectors
that should sum to ~1 — a spot->cell-type mapping, a Leiden->computed-state
merge matrix, etc. Shared between reference_aligners/mapping_analysis and
analysis so both use identical math.
"""

from __future__ import annotations

import numpy as np

# Dominance thresholds reported by the "fraction of rows >= threshold" plot —
# see metrics.onehot_plots.plot_dominance_thresholds.
DOMINANCE_THRESHOLDS = (
    0.99,
    0.95,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
    0.20,
    0.10,
    0.0,
)


def onehot_metrics(mapping: np.ndarray) -> dict:
    """
    Per-row "how one-hot is this row" metrics for a soft assignment matrix.

    Args:
        mapping: Soft row-stochastic matrix (n_rows x n_cols), rows ~sum to 1.

    Returns:
        dict with:
            n_rows, n_cols
            max_prob, gini_impurity, entropy: per-row arrays (n_rows,)
            summary: mean/median/std of each, plus the fraction of rows with
                     max_prob above 0.5 / 0.9 / 0.99.
    """
    mapping = np.clip(np.asarray(mapping, dtype=np.float64), 0.0, None)
    row_sums = mapping.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p = mapping / row_sums  # re-normalize defensively

    n_rows, n_cols = p.shape
    max_prob = p.max(axis=1)
    if n_cols > 1:
        gini_impurity = (1.0 - np.sum(p * p, axis=1)) / (1.0 - 1.0 / n_cols)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(
                n_cols
            )
    else:
        gini_impurity = np.zeros(n_rows)
        entropy = np.zeros(n_rows)

    def _summary(values: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "max_prob": max_prob,
        "gini_impurity": gini_impurity,
        "entropy": entropy,
        "summary": {
            "max_prob": _summary(max_prob),
            "gini_impurity": _summary(gini_impurity),
            "entropy": _summary(entropy),
            "frac_max_prob_above_0.5": float(np.mean(max_prob > 0.5)),
            "frac_max_prob_above_0.9": float(np.mean(max_prob > 0.9)),
            "frac_max_prob_above_0.99": float(np.mean(max_prob > 0.99)),
        },
    }
