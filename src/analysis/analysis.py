"""
Post-mapping analysis for AIM: scores one K's saved mapping outputs against
adata_sc/adata_st and produces the analysis report for that K.

    from analysis.analysis import run_analysis
    run_analysis(adata_sc, adata_st, run_dir)

Reads whatever it needs directly off ``adata_sc``/``adata_st`` (obs[OBS_LEIDEN_ALL_GENES],
uns[UNS_SHARED_GENES / UNS_LEIDEN_RESOLUTION_ALL_GENES / UNS_LEIDEN_SIZES /
UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM / UNS_LEIDEN_EXPR_SUMS_SHARED_GENES*] —
see adata_schema.py) rather than taking it as a separate parameter, and loads
this K's own outputs (spot_to_state_mapping.h5ad onto adata_st.obsm,
leiden_to_state.csv) from ``run_dir``.

Computes, for one K:
    - One-hotness metrics/plots for spot_to_state_mapping.h5ad (P).
    - Hard (argmax) mapping: for every spot, the index of its winning state —
      stored on adata_st.obsm[OBSM_MAPPING_HARD] alongside the soft P.
    - Reconstruction cosine similarity (soft/hard x raw/normalized+log1p),
      via state centroids assembled from the Leiden-cluster expression sums.
    - Biology metrics: spatial organisation of the mapped spots + substate
      merge coherence (both permutation-tested).
    - Hard assignments, fractions, modularity of the computed-state
      partition, and the plots (UMAP grid, spatial map, state profiles,
      leiden-merge map).

Outputs written to run_dir/analysis/:
    data/cossim/, cossim_summary.csv reconstruction cosine similarity
    data/biology_metrics.json        spatial organisation + substate coherence detail
    plots/cell_state_profiles.png    per-state expression heatmap + cell/spot fractions
    plots/cell_state_fractions.png   standalone cell/spot fraction bar charts
    plots/spatial_cell_states.png    spatial plot coloured by computed state
    plots/umap_computed_state.png    computed-state UMAP (shown beside the spatial plot)
    plots/umap_grid.png              2x2 UMAP grid: {Leiden, computed} x {all, shared genes}
    plots/leiden_merge_map.png       which Leiden overclusters merged into each AIM state
    report.pdf                       the PDF report (via analysis.report)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from adata_schema import (
    LAYER_LOGNORM,
    OBS_LEIDEN_ALL_GENES,
    OBS_LEIDEN_SHARED_GENES,
    OBSM_MAPPING_SOFT,
    OBSM_SPATIAL,
    OBSP_CONNECTIVITIES_SHARED_GENES,
    UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM,
    UNS_LEIDEN_EXPR_SUMS_SHARED_GENES,
    UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM,
    UNS_LEIDEN_RESOLUTION_ALL_GENES,
    UNS_LEIDEN_SIZES,
    UNS_SHARED_GENES,
    OBSM_MAPPING_HARD,
)
from metrics.cossim import CossimResult, compute_and_save_cossim
from metrics.cossim_plots import plot_cossim_boxplots

from .utils import hard_assignments
from .assignments import cell_state_fractions
from .biology_metrics import compute_spatial_organization, compute_substate_coherence
from .clustering import compute_modularity
from .mapping_metrics import (
    assemble_state_centroids,
    load_mapping,
    load_leiden_to_state,
    predict_expression,
)
from .onehot import save_onehot
from .plots import (
    _build_state_palette,
    plot_umap_grid,
    plot_leiden_merge_map,
    plot_umap_comparison,
    plot_state_profiles,
    plot_state_fractions,
    plot_spatial_cell_states,
)
from .report import generate_analysis_report

logger = logging.getLogger(__name__)


def run_analysis(adata_sc: AnnData, adata_st: AnnData, run_dir: Path) -> None:
    """
    Score one K's saved mapping outputs against ``adata_sc``/``adata_st``:
    one-hotness metrics, hard (argmax) mapping, reconstruction cosine
    similarity, biology metrics (spatial organisation + substate merge
    coherence), hard assignments/fractions/modularity of the computed-state
    partition, and the UMAP/spatial/state-profile plots — then writes the
    PDF report.

    Args:
        adata_sc: Single-cell AnnData (cells x genes), raw counts in .X;
                  carries obs[OBS_LEIDEN_ALL_GENES / OBS_LEIDEN_SHARED_GENES],
                  uns[UNS_SHARED_GENES / UNS_LEIDEN_RESOLUTION_ALL_GENES /
                  UNS_LEIDEN_SIZES / UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM /
                  UNS_LEIDEN_EXPR_SUMS_SHARED_GENES*] (see adata_schema.py) —
                  written by aim.clustering / aim.aggregation before the K
                  sweep runs, and read back here rather than recomputed. The
                  Leiden overclustering used to train this run is read from
                  here, not recomputed: scanpy's Leiden isn't deterministic
                  run-to-run, so a fresh clustering wouldn't reproduce the
                  actual partition the model was trained on.
        adata_st: Spatial AnnData (spots x genes), carrying layers[LAYER_LOGNORM].
                  Mutated in place: this K's mapping is loaded onto
                  adata_st.obsm[OBSM_MAPPING_SOFT / OBSM_MAPPING_HARD].
        run_dir: one K_<kkk> sweep folder, containing spot_to_state_mapping.h5ad
                  and leiden_to_state.csv (as written by main.py). analysis/ is
                  written inside this folder.
    """

    # Get base paths
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    data_dir = analysis_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load this K's prob spot to state mapping (onto adata_st.obsm) and tree cut (leiden to state)
    load_mapping(run_dir, adata_st)
    leiden_to_state = load_leiden_to_state(run_dir)  # (L,)

    P = adata_st.obsm[OBSM_MAPPING_SOFT]
    k = P.shape[1]

    # Create hard mapping: for every spot, the index of its winning (argmax)
    # state (S,), reshaped to (S x 1) since obsm entries are matrices — stored
    # alongside the soft mapping. spot_states_hard itself is reused below for
    # spatial organisation and the "hard" cossim reconstruction combos.
    logger.info("Computing hard (argmax) mapping...")
    spot_states_hard = P.argmax(axis=1)
    adata_st.obsm[OBSM_MAPPING_HARD] = spot_states_hard.reshape(-1, 1)

    # ── 1. One-hotness — spot_to_state_mapping ──────────────────────────────
    logger.info("Computing one-hot metrics...")
    save_onehot(adata_st, plots_dir, data_dir)

    # ── 2b. Spatial organisation of the mapped spots ────────────────────────
    logger.info("Computing spatial organisation of mapped spots...")
    coords = (
        np.asarray(adata_st.obsm[OBSM_SPATIAL])
        if OBSM_SPATIAL in adata_st.obsm
        else None
    )
    spatial = compute_spatial_organization(spot_states_hard, coords)

    shared_genes = list(adata_sc.uns[UNS_SHARED_GENES])
    sizes = adata_sc.uns[UNS_LEIDEN_SIZES]

    # ── 3. Reconstruction cosine similarity (soft/hard P x raw/norm) ────────
    # shared_genes is guaranteed non-empty here: aim.sweep asserts it before
    # the K sweep (and this analysis) ever runs.
    logger.info("Computing reconstruction cosine similarities...")
    adata_st_shared = adata_st[:, shared_genes]  # view, not a copy
    adata_st_norm = AnnData(
        X=np.asarray(adata_st_shared.layers[LAYER_LOGNORM]),
        obs=pd.DataFrame(index=adata_st_shared.obs_names),
        var=pd.DataFrame(index=shared_genes),
    )

    # Substate merge coherence — on the normalized+log1p shared-gene
    # Leiden-cluster centroids (mean expression per cluster).
    logger.info("Computing substate merge coherence...")
    coherence = compute_substate_coherence(
        adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM], leiden_to_state
    )

    # leiden_to_state is the only merge there is (the tree cut has no soft
    # form) — assemble each centroid set once and reuse it for both the
    # soft-P and hard-P cossim combos below, instead of recomputing it twice.
    centroids_raw = assemble_state_centroids(
        leiden_to_state, k, adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES], sizes
    )
    centroids_norm = assemble_state_centroids(
        leiden_to_state, k, adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM], sizes
    )

    # "hard" combos use spot_states_hard (winning-state index per spot) to
    # gather each spot's assigned-state centroid row directly — no one-hot
    # matrix needed, unlike "soft" which is a genuine mapping @ centroids.
    spot_names = adata_st.obs_names.tolist()
    combos = {
        "soft-raw": (P, centroids_raw, adata_st_shared, False),
        "hard-raw": (spot_states_hard, centroids_raw, adata_st_shared, True),
        "soft-norm": (P, centroids_norm, adata_st_norm, False),
        "hard-norm": (spot_states_hard, centroids_norm, adata_st_norm, True),
    }
    cossim_summary: dict[str, dict] = {}
    cossim_dir = data_dir / "cossim"
    cossim_results: dict[str, CossimResult] = {}
    for label, (m, centroids, st_ref, is_hard) in combos.items():
        pred = (
            centroids[m] if is_hard else predict_expression(m, centroids)
        )  # S x G_shared
        pred_adata = AnnData(
            X=pred.T.astype(np.float32),
            obs=pd.DataFrame(index=shared_genes),
            var=pd.DataFrame(index=spot_names),
        )
        result = compute_and_save_cossim(
            st_ref, pred_adata, cossim_dir, suffix=f"-{label}"
        )
        cossim_results[label] = result
        cossim_summary[label] = {
            "median_gene": result.median_gene,
            "median_spot": result.median_spot,
        }
    pd.DataFrame(cossim_summary).T.to_csv(data_dir / "cossim_summary.csv")
    plot_cossim_boxplots(cossim_results, plots_dir / "cossim_boxplots.png")

    # ── 4. UMAP/modularity/contingency/state-profile pipeline ──────────────
    leiden_idx = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    cell_state_soft = np.zeros(
        (len(leiden_idx), k), dtype=np.float64
    )  # (n_cells x K) cell -> state
    cell_state_soft[np.arange(len(leiden_idx)), leiden_to_state[leiden_idx]] = 1.0
    cell_state_soft[cell_state_soft < 0.1] = 0.0
    spot_state_soft = P.copy()  # (S x K) spot -> state
    spot_state_soft[spot_state_soft < 0.1] = 0.0

    cell_states = hard_assignments(cell_state_soft)
    spot_states = hard_assignments(spot_state_soft)
    state_palette = _build_state_palette(sorted(np.unique(cell_states).tolist()))

    cell_fractions = cell_state_fractions(cell_states, k)
    spot_fractions = cell_state_fractions(spot_states, k)

    # ── Modularity for the computed assignment ──────────────────────────────
    # Two graphs, both precomputed on adata_sc: all genes (default obsp key)
    # and shared genes (OBSP_CONNECTIVITIES_SHARED_GENES) — this only scores
    # the (K-dependent) computed-state partition on the cached graphs.
    # modularity_shared_leiden is the K-independent baseline: modularity of
    # the raw shared-gene Leiden overclustering itself, before any merging.
    logger.info("Computing modularity for computed assignment...")
    modularity_all = compute_modularity(adata_sc, cell_states)
    modularity_shared = compute_modularity(
        adata_sc, cell_states, obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES
    )
    modularity_shared_leiden = compute_modularity(
        adata_sc,
        adata_sc.obs[OBS_LEIDEN_SHARED_GENES].astype(int).to_numpy(),
        obsp_key=OBSP_CONNECTIVITIES_SHARED_GENES,
    )
    logger.info(
        "Modularity: all=%.4f shared=%.4f | shared Leiden ref=%.4f",
        modularity_all,
        modularity_shared,
        modularity_shared_leiden,
    )

    adata_sc.obs["computed_state"] = pd.Categorical(cell_states.astype(str))

    # ── Computed-state UMAP (standalone, for side-by-side with the spatial plot)
    plot_umap_comparison(
        adata_sc,
        panels=[("computed_state", "Computed cell-state assignment")],
        output_path=plots_dir / "umap_computed_state.png",
        state_palette=state_palette,
    )

    # ── UMAP grid: {Leiden all-gene, Leiden shared-gene, computed} x {all, shared}
    plot_umap_grid(
        adata_sc,
        output_path=plots_dir / "umap_grid.png",
        leiden_resolution=float(adata_sc.uns[UNS_LEIDEN_RESOLUTION_ALL_GENES]),
        state_palette=state_palette,
        modularity_all=modularity_all,
        modularity_shared=modularity_shared,
    )

    # ── Spatial cell-state plot ──────────────────────────────────────────────
    plot_spatial_cell_states(
        adata_st,
        spot_states,
        output_path=plots_dir / "spatial_cell_states.png",
        state_palette=state_palette,
    )

    # ── Cell-state profiles ──────────────────────────────────────────────────
    logger.info("Plotting cell-state profiles...")
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

    # ── Leiden-merge map: which Leiden overclusters merged into each state ───
    plot_leiden_merge_map(
        leiden_idx,
        cell_states,
        plots_dir / "leiden_merge_map.png",
        state_palette=state_palette,
    )

    # Persist the biology metrics BEFORE the report is generated — the report's
    # spatial/coherence sections read biology_metrics.json from disk, so it must
    # already exist when generate_analysis_report runs.
    with open(data_dir / "biology_metrics.json", "w") as f:
        json.dump({"spatial": spatial, "coherence": coherence}, f, indent=2)

    generate_analysis_report(analysis_dir)
    logger.info("Analysis report written to %s", analysis_dir)
