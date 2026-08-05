"""Combined per-K scores for the "Comparing K" card.

Each of the three K-sweep criteria is measured by *two* curves over K:

* reconstruction — per-spot and per-gene cosine similarity (raw, hard),
* spatial organisation — neighbourhood-enrichment and local-purity z-score,
* transcriptional coherence — ST-expression and SC-shared-gene modularity.

To plot the criteria against each other, each pair is reduced to one number per
K: their **harmonic mean**, which only rewards a K when *both* of its curves are
high (with an arithmetic mean one large value can carry a small one).

The two spatial z-scores live on very different scales (tens vs. hundreds), so
each is first divided by its own maximum over the sweep — putting both on a
common 0..1 scale — before the mean is taken. Scaling by the maximum (rather
than min-max) keeps zero at zero, so the shuffle null stays where it is. The
other two criteria's curves are already commensurate (cosine similarity,
modularity) and are combined as they are.

Only **reconstruction** carries a label-shuffle null — a chance-level crosshair in
the scatter panels — measured by the sweep (the post-mapping analysis writes the
shuffled-label median next to each observed value and ``k_comparison.csv`` collects
it) and reduced the same way as the score so the two are comparable. Nothing here
computes a metric. The other two criteria draw no crosshair:

* spatial organisation — its curves are already z-scores against exactly that
  shuffle, so the null is 0 for every K; a line pinned at zero says nothing, so it
  is not drawn,
* transcriptional coherence — a shuffled modularity is meaningless, so the sweep
  measures none.

Their null columns are therefore absent and ``combined_null`` yields NaN. The
coherence scatter axis is instead pinned to start at 0 (``axis_from_zero``) so the
modularities are read in absolute terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NULL_SUFFIX = "_null"


@dataclass(frozen=True)
class Criterion:
    """One K-sweep criterion: two measured curves and how to combine them."""

    key: str  # column name of the combined score in the score table
    label: str  # axis / card title
    short: str  # compact label for the scatter-panel titles
    columns: tuple[str, str]  # the two ``k_comparison`` columns
    curve_labels: tuple[str, str]  # legend labels for those columns
    unit: str  # y-axis title of the per-criterion line card
    scale_to_max: bool = False  # divide each curve by its own max over K first
    axis_from_zero: bool = False  # pin this criterion's scatter axis to start at 0


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="recon",
        label="Reconstruction cossim",
        short="Reconstruction",
        columns=("cossim_hard_raw_spot", "cossim_hard_raw_gene"),
        curve_labels=("raw · spot", "raw · gene"),
        unit="cosine similarity",
    ),
    Criterion(
        key="spatial",
        label="Spatial organisation",
        short="Spatial org.",
        columns=("nhood_mean_self_zscore", "local_purity_zscore"),
        curve_labels=("nhood enrichment z", "local purity z"),
        unit="z-score ÷ max over K",
        scale_to_max=True,
    ),
    Criterion(
        key="coherence",
        label="Transcriptional coherence",
        short="Coherence",
        columns=("modularity_st_expression", "modularity_shared"),
        curve_labels=("ST expression (mapping)", "SC shared-gene (reference)"),
        unit="modularity",
        axis_from_zero=True,
    ),
)

CRITERION = {c.key: c for c in CRITERIA}

# The three criterion-vs-criterion scatter panels (x, y).
SCATTER_PAIRS: tuple[tuple[str, str], ...] = (
    ("recon", "spatial"),
    ("recon", "coherence"),
    ("spatial", "coherence"),
)


def harmonic_mean(a, b) -> np.ndarray:
    """Element-wise harmonic mean ``2ab / (a + b)`` of two non-negative arrays.

    A zero in either input gives 0 (the limit of the harmonic mean, so a null of
    zero stays zero); a negative or NaN input gives NaN, leaving a gap rather
    than a meaningless value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    total = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, 2.0 * a * b / np.where(total > 0, total, 1.0), 0.0)
    bad = ~np.isfinite(a) | ~np.isfinite(b) | (a < 0) | (b < 0)
    return np.where(bad, np.nan, out)


def _column(df: pd.DataFrame, column: str, n: int) -> np.ndarray:
    """``df[column]`` as a float array, or all-NaN when the column is absent."""
    if column not in df.columns:
        return np.full(n, np.nan)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def scale_factors(df: pd.DataFrame, criterion: Criterion) -> tuple[float, float]:
    """Per-curve factor that puts both curves of ``criterion`` on one scale.

    ``1.0`` for the criteria whose curves are already commensurate; ``1 / max``
    over the sweep otherwise. A curve whose maximum is not finite and positive
    yields NaN, which propagates to the combined score for that criterion.
    """
    if not criterion.scale_to_max:
        return (1.0, 1.0)
    factors = []
    for column in criterion.columns:
        values = _column(df, column, len(df))
        finite = values[np.isfinite(values)]
        top = float(finite.max()) if finite.size else float("nan")
        factors.append(1.0 / top if np.isfinite(top) and top > 0 else float("nan"))
    return (factors[0], factors[1])


def scaled_curves(
    df: pd.DataFrame, criterion: Criterion
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(raw_a, raw_b, scaled_a, scaled_b)`` for one criterion's two curves."""
    n = len(df)
    raw = [_column(df, column, n) for column in criterion.columns]
    fa, fb = scale_factors(df, criterion)
    return (raw[0], raw[1], raw[0] * fa, raw[1] * fb)


def combined(df: pd.DataFrame, criterion: Criterion) -> np.ndarray:
    """The criterion's combined score per K (harmonic mean of its scaled curves)."""
    *_, scaled_a, scaled_b = scaled_curves(df, criterion)
    return harmonic_mean(scaled_a, scaled_b)


def combined_null(df: pd.DataFrame, criterion: Criterion) -> np.ndarray:
    """The criterion's combined label-shuffle null per K, or all-NaN when it has
    none (then no crosshair is drawn).

    The null is the harmonic mean of the ``<column>_null`` columns the sweep
    collected, scaled with the same factors as the observed curves. Only
    reconstruction has those columns; spatial (null always 0) and coherence (a
    shuffled modularity is meaningless) do not, so they come through as NaN.
    """
    n = len(df)
    fa, fb = scale_factors(df, criterion)
    values = [
        _column(df, f"{column}{NULL_SUFFIX}", n) * factor
        for column, factor in zip(criterion.columns, (fa, fb))
    ]
    return harmonic_mean(values[0], values[1])


def pareto_mask(table: pd.DataFrame, keys: tuple[str, ...] | None = None) -> np.ndarray:
    """Boolean mask of the Pareto-optimal rows of a score table.

    Each K is a point in the space spanned by ``keys`` (by default all three
    criteria's combined scores) and every criterion is maximised, so a K drops out
    only when some other K is at least as good on *all* of them and strictly
    better on one. A missing (NaN) score counts as the worst possible value, so a
    K with an uncomputable criterion survives only by leading on another one.
    """
    columns = keys or tuple(c.key for c in CRITERIA)
    n = len(table)
    if n == 0:
        return np.zeros(0, dtype=bool)
    points = np.column_stack(
        [_column(table, column, n) for column in columns]
    )  # (n, dims)
    points = np.where(np.isfinite(points), points, -np.inf)

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        others = np.delete(points, i, axis=0)
        if others.size == 0:
            continue
        dominated = np.all(others >= points[i], axis=1) & np.any(
            others > points[i], axis=1
        )
        keep[i] = not dominated.any()
    return keep


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per K with each criterion's combined score and its shuffle null.

    ``df`` is the sweep table (``k_comparison.csv``), which carries both the
    measured curves and their nulls. Columns: ``k``, then ``<key>`` and
    ``<key>_null`` for every criterion in :data:`CRITERIA`.
    """
    out = pd.DataFrame({"k": df["k"].astype(int).to_numpy()})
    for criterion in CRITERIA:
        out[criterion.key] = combined(df, criterion)
        out[f"{criterion.key}{NULL_SUFFIX}"] = combined_null(df, criterion)
    return out
