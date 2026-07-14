"""
Post-mapping analysis for AIM.

Standalone entry point; call run_analysis() after mapping:

    from analysis.analysis import run_analysis
    results = run_analysis(adata_sc, adata_st, cell_state_soft, spot_state_soft, output_dir=Path("analysis_out"), leiden_resolution=3.0, n_leiden=5, leiden_labels=leiden_labels)

Outputs written to output_dir:
    cell_state_profiles.png          per-state expression heatmap + cell/spot fractions
    cell_state_fractions.png         standalone cell/spot fraction bar charts
    spatial_cell_states.png          spatial plot coloured by computed state
    umap_computed_state.png          computed-state UMAP (shown beside the spatial plot)
    umap_allgenes_vs_shared.png      computed-state UMAP: all-gene vs shared-gene embedding
    umap_comparison.png              UMAP: computed assignment vs Leiden (shared & all genes)
    contingency_heatmap.png          contingency matrix (Leiden, all genes)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData

from utils import _to_numpy, run_pca_neighbors_umap, hard_assignments
from .assignments import cell_state_fractions
from .clustering import (
    compute_modularity,
    run_leiden_shared_genes,
)
from .matching import (
    compute_contingency_matching,
    plot_contingency_heatmap,
)
from .plots import (
    _build_state_palette,
    plot_computed_state_umaps,
    plot_umap_comparison,
    plot_state_profiles,
    plot_state_fractions,
    plot_spatial_cell_states,
)

logger = logging.getLogger(__name__)


def run_analysis(
    adata_sc: AnnData,
    adata_st: AnnData,
    cell_state_soft: torch.Tensor | np.ndarray,
    spot_state_soft: torch.Tensor | np.ndarray,
    output_dir: Path,
    leiden_resolution: float,
    n_leiden: int,
    leiden_labels: np.ndarray,
) -> dict:
    """
    Full post-mapping analysis pipeline.

    Parameters
    ----------
    adata_sc         : Single-cell AnnData (cells × genes), raw counts expected.
    adata_st         : Spatial AnnData (spots × genes).
    cell_state_soft  : Cell-to-state soft-assignment matrix (n_cells, n_leiden).
    spot_state_soft  : Spot-to-state soft-assignment matrix (n_spots, n_leiden).
    output_dir       : Directory where all outputs are written (created if absent).
    n_leiden         : Number of AIM state slots — equal to L, the number of
                       Leiden overclustering clusters (AIM's G matrix is L x L,
                       so L is both the Leiden-cluster count and the total
                       number of state slots; see model.py). Not every slot is
                       necessarily used — see n_computed_states/n_mapped_states
                       below for the actually-occupied count.
    leiden_resolution     : Resolution for the main Leiden reference clusterings.
    leiden_labels    : Leiden cluster id per cell (n_cells,), integer — the exact
                       "all genes" overclustering used to train this run (e.g.
                       loaded from leiden_overclustering.h5ad). Not recomputed
                       here: scanpy's Leiden isn't deterministic run-to-run, so a
                       fresh clustering wouldn't reproduce the actual partition
                       the model was trained on.

    Returns
    -------
    dict with keys:
        cell_states            np.ndarray (n_cells,)   hard cell-state labels
        spot_states            np.ndarray (n_spots,)   hard cell-state labels
        cell_fractions         dict[int, float]         fraction of cells per state
        spot_fractions         dict[int, float]         fraction of spots per state
        n_mapped_states_above_1pct int                   mapped states with >1% of spots
        metrics_computed       dict[str, float]         modularity (all genes) and
                                                        modularity_shared (shared-gene graph)
        modularity_shared_leiden float                  shared-gene modularity of the
                                                        Leiden-shared partition (≈ ceiling)
        contingency_matching dict   contingency argmax matching (Leiden, all genes)
        adata_processed        AnnData           sc data with UMAP + all labels
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    cell_state_soft = _to_numpy(cell_state_soft)
    spot_state_soft = _to_numpy(spot_state_soft)

    shared_genes = list(set(adata_sc.var_names) & set(adata_st.var_names))
    logger.info(
        "run_analysis: n_leiden=%d, cells=%d, spots=%d, shared_genes=%d",
        n_leiden,
        len(adata_sc),
        len(adata_st),
        len(shared_genes),
    )

    # ── Hard assignments ──────────────────────────────────────────────────────
    cell_states = hard_assignments(cell_state_soft)
    spot_states = hard_assignments(spot_state_soft)
    state_palette = _build_state_palette(sorted(np.unique(cell_states).tolist()))

    n_computed_states = int(len(np.unique(cell_states)))
    n_mapped_states = int(len(np.unique(spot_states)))

    # ── Fractions ─────────────────────────────────────────────────────────────
    cell_fractions = cell_state_fractions(cell_states, n_leiden)
    spot_fractions = cell_state_fractions(spot_states, n_leiden)

    # ── Prepare sc data ────────────────────────────────────────────────────────
    adata_processed = adata_sc.copy()
    run_pca_neighbors_umap(adata_processed)

    # ── Leiden reference – shared genes ───────────────────────────────────────
    # Keep the shared-gene AnnData (with its KNN graph) for the shared modularity.
    leiden_shared_labels, adata_shared = run_leiden_shared_genes(
        adata_sc, shared_genes=shared_genes, resolution=leiden_resolution
    )

    # ── Modularity for the computed assignment ────────────────────────────────
    # Two graphs: all genes (existing) and shared genes only — the latter is the
    # space the method actually operates in (ST is only seen through shared genes).
    logger.info("Computing modularity for computed assignment…")
    metrics_computed = {
        "modularity": compute_modularity(adata_processed, cell_states),
        "modularity_shared": compute_modularity(adata_shared, cell_states),
    }
    # Leiden-shared partition on the same graph ≈ ceiling (Leiden ~maximises Q)
    modularity_shared_leiden = compute_modularity(adata_shared, leiden_shared_labels)
    logger.info(
        "Modularity: %s | shared Leiden ref=%.4f",
        metrics_computed,
        modularity_shared_leiden,
    )

    adata_processed.obs["computed_state"] = pd.Categorical(cell_states.astype(str))
    adata_processed.obs["leiden_state"] = pd.Categorical(leiden_labels.astype(str))
    adata_processed.obs["leiden_shared_state"] = pd.Categorical(
        leiden_shared_labels.astype(str)
    )

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
        state_palette=state_palette,
    )

    # ── Computed-state UMAP (standalone, for side-by-side with the spatial plot)
    plot_umap_comparison(
        adata_processed,
        panels=[("computed_state", "Computed cell-state assignment")],
        output_path=plots_dir / "umap_computed_state.png",
        state_palette=state_palette,
    )

    # ── Computed states: all-gene vs shared-gene UMAP (side by side) ──────────
    adata_shared.obs["computed_state"] = pd.Categorical(cell_states.astype(str))
    if "X_umap" not in adata_shared.obsm:
        sc.tl.umap(adata_shared)
    plot_computed_state_umaps(
        adata_processed,
        adata_shared,
        output_path=plots_dir / "umap_allgenes_vs_shared.png",
        state_palette=state_palette,
        modularity_all=metrics_computed["modularity"],
        modularity_shared=metrics_computed["modularity_shared"],
    )

    # ── Spatial cell-state plot ───────────────────────────────────────────────
    plot_spatial_cell_states(
        adata_st,
        spot_states,
        output_path=plots_dir / "spatial_cell_states.png",
        state_palette=state_palette,
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
    n_mapped_states_above_1pct = int(
        sum(1 for f in spot_fractions.values() if f > 0.01)
    )

    return {
        "cell_states": cell_states,
        "spot_states": spot_states,
        "n_computed_states": n_computed_states,
        "n_computed_states_above_1pct": n_computed_states_above_1pct,
        "n_mapped_states": n_mapped_states,
        "n_mapped_states_above_1pct": n_mapped_states_above_1pct,
        "cell_fractions": cell_fractions,
        "spot_fractions": spot_fractions,
        "metrics_computed": metrics_computed,
        "modularity_shared_leiden": modularity_shared_leiden,
        "contingency_matching": contingency_matching,
        "adata_processed": adata_processed,
    }
