"""
Post-mapping analysis for AlternativeIdea.

Standalone entry point; call run_analysis() after mapping:

    from src.alternative_idea.src.evaluate_k.analysis import run_analysis
    results = run_analysis(adata_sc, adata_st, B, C, output_dir=Path("analysis_out"), K=5)

Outputs written to output_dir:
    cell_state_profiles.png          per-state expression heatmap + cell/spot fractions
    cell_state_fractions.png         standalone cell/spot fraction bar charts
    spatial_cell_states.png          spatial plot coloured by computed state
    umap_comparison.png              UMAP: computed assignment vs Leiden (shared & all genes)
    contingency_heatmap.png          contingency matrix (Leiden, all genes)
    substate_metrics.csv             per-state sub-cluster metrics
    crosstab_heatmap.png             predicted state × GT cell type (only if gt_label_key given)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from ..utils import _to_numpy, run_pca_neighbors_umap, hard_assignments
from .assignments import cell_state_fractions
from .clustering import (
    compute_modularity,
    run_leiden_clustering,
    run_leiden_shared_genes,
)
from .matching import (
    compute_contingency_matching,
    plot_contingency_heatmap,
)
from .plots import (
    _build_state_palette,
    plot_umap_comparison,
    plot_state_profiles,
    plot_state_fractions,
    plot_spatial_cell_states,
)
from .substate_analysis import compute_substate_metrics

logger = logging.getLogger(__name__)


def run_analysis(
    adata_sc: AnnData,
    adata_st: AnnData,
    B: torch.Tensor | np.ndarray,
    C: torch.Tensor | np.ndarray,
    output_dir: Path,
    leiden_resolution: float,
    K: int,
    adata_processed_base: AnnData | None = None,
    leiden_labels: np.ndarray | None = None,
    leiden_shared_labels: np.ndarray | None = None,
) -> dict:
    """
    Full post-mapping analysis pipeline.

    Parameters
    ----------
    adata_sc         : Single-cell AnnData (cells × genes), raw counts expected.
    adata_st         : Spatial AnnData (spots × genes).
    B                : Cell-to-state soft-assignment matrix (n_cells, K).
    C                : Spot-to-state soft-assignment matrix (n_spots, K).
    output_dir       : Directory where all outputs are written (created if absent).
    K                : Number of cell states (inferred from B if None).
    leiden_resolution     : Resolution for the main Leiden reference clusterings.

    Returns
    -------
    dict with keys:
        cell_states            np.ndarray (n_cells,)   hard cell-state labels
        spot_states            np.ndarray (n_spots,)   hard cell-state labels
        cell_fractions         dict[int, float]         fraction of cells per state
        spot_fractions         dict[int, float]         fraction of spots per state
        metrics_computed       dict[str, float]         modularity for computed assignment
        contingency_matching dict   contingency argmax matching (Leiden, all genes)
        substate_metrics     dict   per-state sub-cluster analysis results
        adata_processed        AnnData           sc data with UMAP + all labels
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    B = _to_numpy(B)
    C = _to_numpy(C)

    shared_genes = list(set(adata_sc.var_names) & set(adata_st.var_names))
    logger.info(
        "run_analysis: K=%d, cells=%d, spots=%d, shared_genes=%d",
        K,
        len(adata_sc),
        len(adata_st),
        len(shared_genes),
    )

    # ── Hard assignments ──────────────────────────────────────────────────────
    cell_states = hard_assignments(B)
    spot_states = hard_assignments(C)
    state_palette = _build_state_palette(sorted(np.unique(cell_states).tolist()))

    n_computed_states = int(len(np.unique(cell_states)))
    n_mapped_states = int(len(np.unique(spot_states)))

    # ── Fractions ─────────────────────────────────────────────────────────────
    cell_fractions = cell_state_fractions(cell_states, K)
    spot_fractions = cell_state_fractions(spot_states, K)

    # ── Prepare sc data (copy pre-computed base or compute from scratch) ──────
    if adata_processed_base is not None:
        adata_processed = adata_processed_base.copy()
    else:
        adata_processed = adata_sc.copy()
        run_pca_neighbors_umap(adata_processed)

    # ── Leiden reference – all genes (pre-computed or on demand) ─────────────
    if leiden_labels is None:
        leiden_labels, _ = run_leiden_clustering(adata_sc, resolution=leiden_resolution)

    # ── Leiden reference – shared genes (pre-computed or on demand) ───────────
    if leiden_shared_labels is None:
        leiden_shared_labels, _ = run_leiden_shared_genes(
            adata_sc, shared_genes=shared_genes, resolution=leiden_resolution
        )

    # ── Modularity for the computed assignment ────────────────────────────────
    logger.info("Computing modularity for computed assignment…")
    metrics_computed = {"modularity": compute_modularity(adata_processed, cell_states)}
    logger.info("Computed modularity: %s", metrics_computed)

    adata_processed.obs["computed_state"] = pd.Categorical(cell_states.astype(str))
    adata_processed.obs["leiden_state"] = pd.Categorical(leiden_labels.astype(str))
    adata_processed.obs["leiden_shared_state"] = pd.Categorical(
        leiden_shared_labels.astype(str)
    )

    # ── Per-state sub-cluster analysis ───────────────────────────────────────
    logger.info("Computing per-state sub-cluster metrics…")
    substate = compute_substate_metrics(
        adata_sc,
        adata_st,
        cell_states,
        spot_states,
        leiden_labels,
        shared_genes,
        n_top_genes=30,
        n_perm=100,
    )

    def _fmt_row(cs: int, metrics: dict) -> dict:
        def _f(v):
            try:
                return v if v == v else "—"  # NaN != NaN
            except TypeError:
                return v

        return {
            "CS": int(cs),
            "cells": int(metrics["n_cells"]),
            "spots": int(metrics["n_spots"]),
            "n_LS": int(metrics["n_leiden_sub"]),
            "cosim": _f(metrics["cossim_centroid"]),
            "perm_p": _f(metrics["perm_p_value"]),
        }

    pd.DataFrame(
        [_fmt_row(cs, m) for cs, m in sorted(substate["per_state"].items())]
    ).to_csv(data_dir / "substate_metrics.csv", index=False)

    logger.info("Sub-cluster metrics → %s/substate_metrics.csv", data_dir)

    # ── Combined UMAP comparison ──────────────────────────────────────────────
    plot_umap_comparison(
        adata_processed,
        panels=[
            ("computed_state", "Computed cell-state assignment"),
            ("leiden_state", f"Leiden – all genes (resolution={leiden_resolution})"),
            (
                "leiden_shared_state",
                f"Leiden – shared genes (resolution={leiden_resolution})",
            ),
        ],
        output_path=plots_dir / "umap_comparison.png",
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
        state_palette=state_palette,
    )

    # ── Spatial cell-state plot ───────────────────────────────────────────────
    plot_spatial_cell_states(
        adata_st,
        spot_states,
        output_path=plots_dir / "spatial_cell_states.png",
        state_palette=state_palette,
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
    )

    # ── Cell-state profiles ───────────────────────────────────────────────────
    logger.info("Plotting cell-state profiles…")
    plot_state_profiles(
        adata_sc,
        cell_states,
        shared_genes,
        plots_dir / "cell_state_profiles.png",
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
        state_palette=state_palette,
    )
    plot_state_fractions(
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
        unique_states=sorted(np.unique(cell_states).tolist()),
        output_path=plots_dir / "cell_state_fractions.png",
        state_palette=state_palette,
    )

    # ── Contingency matching (Leiden, all genes) ─────────────────────────────
    logger.info("Computing contingency matching…")
    contingency_matching = compute_contingency_matching(cell_states, leiden_labels)
    plot_contingency_heatmap(
        contingency_matching,
        plots_dir / "contingency_heatmap.png",
        spot_fractions=spot_fractions,
    )

    n_computed_states_above_1pct = int(
        sum(1 for f in cell_fractions.values() if f > 0.01)
    )

    return {
        "cell_states": cell_states,
        "spot_states": spot_states,
        "n_computed_states": n_computed_states,
        "n_computed_states_above_1pct": n_computed_states_above_1pct,
        "n_mapped_states": n_mapped_states,
        "cell_fractions": cell_fractions,
        "spot_fractions": spot_fractions,
        "metrics_computed": metrics_computed,
        "contingency_matching": contingency_matching,
        "substate_metrics": substate,
        "adata_processed": adata_processed,
    }
