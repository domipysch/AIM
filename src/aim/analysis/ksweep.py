"""Cross-K comparison: gather each K's key analysis metrics from its per-K
``analysis/data`` outputs into one table at the run root (``k_comparison.csv``),
score them, and record which K each criterion selects (``k_selection.json``).

``k_comparison.csv`` holds the measured metrics *and* what they reduce to: one
combined score per criterion, the overall score across criteria, and a ``pareto``
flag. Reads the files each K's post-mapping analysis already writes
(``cossim_summary.csv``, ``topology_metrics.json``, ``modularity_metrics.json``),
so it runs after the K-loop with no recomputation — including the label-shuffle
nulls those steps measured, which the GUI needs alongside the observed values.
Metrics missing for a given K come through as NaN. The scoring itself lives in
:mod:`aim.metrics.kselection`, so a CLI run and the GUI's "Comparing K" card
report the same numbers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aim.metrics import kselection

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


def _cossim_cell(cs: pd.DataFrame, row: str, column: str) -> float:
    """One cell of ``cossim_summary.csv`` as a float, NaN when absent."""
    if row not in cs.index or column not in cs.columns:
        return float("nan")
    return _finite(cs.loc[row, column])


def collect_k_metrics(output_folder: Path, ks: list[int]) -> pd.DataFrame:
    """Gather the per-K comparison metrics into one DataFrame (one row per K, ascending).

    Columns: ``k``, ``cossim_hard_norm_spot``, ``cossim_hard_norm_gene``,
    ``cossim_hard_raw_spot``, ``cossim_hard_raw_gene``, ``nhood_mean_self_zscore``,
    ``local_purity_zscore``, ``modularity_shared``, ``modularity_st_expression``,
    ``mean_confidence``, plus the ``*_null`` label-shuffle baseline of every metric
    that has one (the two z-scores are already measured against that shuffle, so
    their null is 0 by construction and is not repeated here).
    """
    rows = []
    for k in ks:
        data_dir = output_folder / f"k_{k:03d}" / "analysis" / "data"

        cossim_norm_spot = cossim_norm_gene = np.nan
        cossim_raw_spot = cossim_raw_gene = np.nan
        cossim_raw_spot_null = cossim_raw_gene_null = np.nan
        cossim_path = data_dir / "cossim_summary.csv"
        if cossim_path.exists():
            cs = pd.read_csv(cossim_path, index_col=0)
            cossim_norm_spot = _cossim_cell(cs, "hard-norm", "median_spot")
            cossim_norm_gene = _cossim_cell(cs, "hard-norm", "median_gene")
            cossim_raw_spot = _cossim_cell(cs, "hard-raw", "median_spot")
            cossim_raw_gene = _cossim_cell(cs, "hard-raw", "median_gene")
            cossim_raw_spot_null = _cossim_cell(cs, "hard-raw", "median_spot_null")
            cossim_raw_gene_null = _cossim_cell(cs, "hard-raw", "median_gene_null")

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
        mod_shared = _finite(modularity.get("modularity_shared", np.nan))
        mod_st = _finite(modularity.get("modularity_st_expression", np.nan))

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
                "cossim_hard_raw_spot_null": cossim_raw_spot_null,
                "cossim_hard_raw_gene_null": cossim_raw_gene_null,
                "nhood_mean_self_zscore": self_z,
                "local_purity_zscore": purity_z,
                "modularity_shared": mod_shared,
                "modularity_st_expression": mod_st,
                "mean_confidence": mean_conf,
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def score_k_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with the K-selection columns appended: one combined score per
    criterion, the ``overall`` score across them, and a ``pareto`` flag.

    The combined scores' own label-shuffle nulls are not carried here — only
    reconstruction has one, and the GUI recomputes it in memory for the chance
    crosshair; the measured nulls it is derived from are already columns of
    ``df``.
    """
    table = kselection.score_table(df)
    out = df.copy()
    for criterion in kselection.CRITERIA:
        out[criterion.key] = table[criterion.key].to_numpy()
    out["overall"] = kselection.overall(table)
    out["pareto"] = kselection.pareto_mask(table)
    return out


def select_k(output_folder: Path, df: pd.DataFrame) -> dict:
    """Write ``k_selection.json``: which K wins, per criterion and overall.

    The proposal is the Pareto-optimal K plus the best K for each criterion and
    for the harmonic mean across all of them, each with its score. Same
    computation the GUI's "Comparing K" card performs, so the CLI no longer
    hides it. Returns the proposal dict.
    """
    table = kselection.score_table(df)
    best = kselection.best_ks(table)
    best_json: dict[str, dict[str, float] | None] = {
        name: (None if hit is None else {"k": hit[0], "score": hit[1]})
        for name, hit in best.items()
    }
    proposal = {
        "pareto_ks": [int(k) for k in table.loc[kselection.pareto_mask(table), "k"]],
        "best": best_json,
        "criteria": {c.key: c.label for c in kselection.CRITERIA},
    }
    json_path = output_folder / "k_selection.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(proposal, fh, indent=2)

    chosen = ", ".join(
        f"{name}={hit[0]}" for name, hit in best.items() if hit is not None
    )
    logger.info("K selection written to %s (%s)", json_path, chosen)
    return proposal


def compare_k_runs(output_folder: Path, ks: list[int]) -> None:
    """Write ``k_comparison.csv`` at the run root: the swept K-runs' reconstruction
    cosine similarity, spatial organisation and mapping modularity, together with
    the criterion scores those reduce to, the overall score and the Pareto flag
    (the GUI renders the comparison plots from it on demand). The winning K per
    criterion and overall goes to ``k_selection.json`` (see :func:`select_k`)."""
    output_folder = Path(output_folder)
    df = collect_k_metrics(output_folder, ks)
    csv_path = output_folder / "k_comparison.csv"
    score_k_metrics(df).to_csv(csv_path, index=False)
    logger.info("K-sweep comparison written to %s", csv_path)
    select_k(output_folder, df)
