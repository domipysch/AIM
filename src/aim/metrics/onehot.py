"""One-hotness metrics for a soft row-stochastic assignment matrix (rows are
probability-like vectors summing to ~1)."""

from __future__ import annotations

import numpy as np


def onehot_metrics(mapping: np.ndarray) -> dict:
    """Per-row one-hotness of a soft assignment matrix (n_rows x n_cols).

    Returns a dict with ``n_rows``, ``n_cols``, the per-row arrays ``max_prob``,
    ``gini_impurity`` and ``entropy``, and a ``summary`` of mean/median/std plus
    the fraction of rows whose max prob exceeds 0.5 / 0.9 / 0.99.
    """
    mapping = np.clip(np.asarray(mapping, dtype=np.float64), 0.0, None)
    row_sums = mapping.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p = mapping / row_sums

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
