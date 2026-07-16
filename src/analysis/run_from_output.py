"""Post-mapping analysis, decoupled from both the AIM runner (main.py) and
grid_search.py: reads a run's saved mapping outputs from disk and produces the
analysis report from them. Neither main.py nor grid_search.py compute the
analysis matrices inline anymore — both call into this module instead.

Computes, for one AIM run:
    - One-hotness metrics/plots for mapping_prob.h5ad (P) and
      leiden_merge_prob.h5ad (G) separately.
    - Hard (argmax) versions of P and G, with a consistency check (raises if
      a spot hard-maps to a state with no Leiden-cluster support).
    - Reconstruction cosine similarity (soft/hard x raw/normalized+log1p),
      via state centroids assembled from the Leiden-cluster expression sums.
    - The existing UMAP/modularity/contingency/state-profile analysis
      (analysis.analysis.run_analysis).

Standalone usage (after `python main.py` has written its outputs):
    python -m analysis.run_from_output \\
        --scdata sc.h5ad --stdata st.h5ad --output_folder <run_dir>

The Leiden resolution is not a parameter here — it's read from the run's own
config.yaml (written by main.py at training time), so the analysis always
matches the resolution actually used to train the run.
"""

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from anndata import AnnData

from metrics.cossim import CossimResult, compute_and_save_cossim
from metrics.cossim_plots import plot_cossim_boxplots
from metrics.onehot import onehot_metrics
from metrics.onehot_plots import plot_dominance_thresholds, plot_onehot_distribution

from .analysis import run_analysis
from .biology_metrics import (
    compute_spatial_organization,
    compute_substate_coherence,
    flatten_biology_objectives,
)
from .mapping_metrics import (
    assemble_state_centroids,
    compute_hard_mapping_validated,
    compute_leiden_expression_sums,
    load_mapping_matrices,
    predict_expression,
    save_matrix_h5ad,
)
from .report import generate_analysis_report

logger = logging.getLogger(__name__)


def _read_leiden_resolution(output_folder: Path) -> float:
    """Read the Leiden resolution used to train this run from its own
    config.yaml (written by main.py) — avoids requiring the caller to
    re-specify a value that could silently mismatch the one actually used."""
    config_path = output_folder / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found in {output_folder} — cannot determine the "
            "Leiden resolution used to train this run."
        )
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    try:
        return float(cfg["reference_leiden_clustering_resolution"])
    except KeyError as e:
        raise KeyError(
            f"config.yaml in {output_folder} has no "
            "training.reference_leiden_clustering_resolution key."
        ) from e


def _save_onehot(
    matrix, key: str, row_label: str, names: list[str], plots_dir: Path, data_dir: Path
) -> dict:
    """Compute + save + plot one-hotness metrics for one matrix (P or G).

    Returns the onehot_metrics dict so its summary can be folded into the flat
    objective_metrics.csv."""
    m = onehot_metrics(matrix)
    pd.DataFrame(
        {
            "id": names,
            "max_prob": m["max_prob"],
            "gini_impurity": m["gini_impurity"],
            "entropy": m["entropy"],
        }
    ).to_csv(data_dir / f"onehot_per_row_{key}.csv", index=False)
    with open(data_dir / f"onehot_summary_{key}.json", "w") as f:
        json.dump(
            {"n_rows": m["n_rows"], "n_cols": m["n_cols"], "summary": m["summary"]},
            f,
            indent=2,
        )
    plot_onehot_distribution(
        m, plots_dir / f"onehot_distribution_{key}.png", row_label=row_label
    )
    plot_dominance_thresholds(
        m, plots_dir / f"onehot_thresholds_{key}.png", row_label=row_label
    )
    return m


_COSSIM_COMBOS = ("soft-raw", "hard-raw", "soft-norm", "hard-norm")


def _collect_objectives(
    results: dict,
    onehot_P: dict,
    onehot_G: dict,
    cossim_summary: dict,
    spatial: dict,
    coherence: dict,
    n_leiden: int,
    n_active_states: int,
    n_mapped_states: int,
) -> dict[str, float]:
    """Flatten every scalar the analysis produces into one grid-search-facing row.

    Columns are emitted in a fixed schema (missing values -> NaN) so the header
    is stable across every run of a grid search, regardless of which optional
    blocks (e.g. cossim when there are no shared genes) actually ran.
    """
    obj: dict[str, float] = {}

    # Mapping sharpness (one-hotness) of P (spot->state) and G (leiden->state).
    for prefix, m in (("sharp_mapping", onehot_P), ("sharp_merge", onehot_G)):
        s = m["summary"]
        obj[f"{prefix}_max_prob_mean"] = float(s["max_prob"]["mean"])
        obj[f"{prefix}_gini_mean"] = float(s["gini_impurity"]["mean"])
        obj[f"{prefix}_frac_above_0.9"] = float(s["frac_max_prob_above_0.9"])

    # Reconstruction cosine similarity (fixed 4 combos x gene/spot).
    for combo in _COSSIM_COMBOS:
        c = cossim_summary.get(combo, {})
        key = combo.replace("-", "_")
        obj[f"recon_{key}_gene"] = float(c.get("median_gene", float("nan")))
        obj[f"recon_{key}_spot"] = float(c.get("median_spot", float("nan")))

    # Modularity + state counts (from run_analysis).
    mc = results.get("metrics_computed", {})
    obj["modularity_all"] = float(mc.get("modularity", float("nan")))
    obj["modularity_shared"] = float(mc.get("modularity_shared", float("nan")))
    obj["modularity_shared_leiden"] = float(
        results.get("modularity_shared_leiden", float("nan"))
    )
    obj["n_leiden"] = float(n_leiden)
    obj["n_active_states"] = float(n_active_states)
    obj["n_mapped_states"] = float(n_mapped_states)
    obj["n_computed_states"] = float(results.get("n_computed_states", float("nan")))
    obj["n_computed_states_above_1pct"] = float(
        results.get("n_computed_states_above_1pct", float("nan"))
    )
    obj["n_mapped_states_above_1pct"] = float(
        results.get("n_mapped_states_above_1pct", float("nan"))
    )

    # Biology: spatial organisation + substate merge coherence.
    obj.update(flatten_biology_objectives(spatial, coherence))
    return obj


def analyze_run(
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
) -> tuple[dict, int, dict]:
    """
    Load one run's saved mapping outputs and run the full post-mapping
    analysis: one-hotness metrics, hard mapping + validation, reconstruction
    cosine similarity, the biology metrics (spatial organisation of the mapped
    spots + substate merge coherence), and the existing UMAP/modularity/
    contingency pipeline.

    Args:
        sc_path, st_path: Full paths to the sc/st h5ad used for the run.
        output_folder:    Folder containing mapping_prob.h5ad, leiden_merge_prob.h5ad,
                           leiden_overclustering.h5ad, and config.yaml (as written by
                           main.py). analysis/ is written inside this folder. The
                           Leiden resolution is read from config.yaml — not passed in
                           — so it always matches the resolution the run was trained
                           with.

    Writes (in analysis/data/): biology_metrics.json (full spatial + coherence
    detail) and objective_metrics.csv (one row of flat scalar objectives — the
    grid-search-facing summary of everything: sharpness, reconstruction,
    modularity, spatial organisation, merge coherence).

    Returns:
        (results, n_leiden, objectives): the dict returned by run_analysis, the
        total number of AIM state slots (= L, the number of Leiden overclustering
        clusters — see model.py), and the flat scalar-objective dict also written
        to objective_metrics.csv.

    Raises:
        ValueError: if the hard mapping is inconsistent (see
                    mapping_metrics.compute_hard_mapping_validated).
    """
    output_folder = Path(output_folder)
    analysis_dir = output_folder / "analysis"
    plots_dir = analysis_dir / "plots"
    data_dir = analysis_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    leiden_resolution = _read_leiden_resolution(output_folder)

    adata_sc = ad.read_h5ad(sc_path)
    adata_st = ad.read_h5ad(st_path)

    logger.info("Loading mapping matrices...")
    P, G, spot_names, leiden_names, state_names, leiden_idx = load_mapping_matrices(
        output_folder
    )
    n_leiden = len(leiden_names)  # = L = len(state_names): total AIM state slots

    # ── 1. One-hotness — mapping_prob (P) and leiden_merge_prob (G) ─────────
    logger.info("Computing one-hot metrics...")
    onehot_P = _save_onehot(P, "mapping_prob", "spot", spot_names, plots_dir, data_dir)
    onehot_G = _save_onehot(
        G, "leiden_merge_prob", "leiden cluster", leiden_names, plots_dir, data_dir
    )

    # ── 2. Hard mapping (argmax), with validation ───────────────────────────
    logger.info("Computing hard (argmax) mapping...")
    P_hard, G_hard, n_active_states, n_mapped_states = compute_hard_mapping_validated(
        P, G
    )
    save_matrix_h5ad(P_hard, spot_names, state_names, data_dir / "mapping_hard.h5ad")
    save_matrix_h5ad(
        G_hard, leiden_names, state_names, data_dir / "leiden_merge_hard.h5ad"
    )

    # ── 2b. Spatial organisation of the mapped spots ────────────────────────
    logger.info("Computing spatial organisation of mapped spots...")
    spot_states = np.asarray(P).argmax(axis=1)
    coords = (
        np.asarray(adata_st.obsm["spatial"]) if "spatial" in adata_st.obsm else None
    )
    spatial = compute_spatial_organization(spot_states, coords)

    # ── 3. Reconstruction cosine similarity (soft/hard x raw/norm) ──────────
    logger.info("Computing reconstruction cosine similarities...")
    shared_genes = sorted(set(adata_sc.var_names) & set(adata_st.var_names))
    cossim_summary: dict[str, dict] = {}
    coherence = compute_substate_coherence(
        np.zeros((n_leiden, 0)), G_hard
    )  # placeholder (no shared genes); overwritten below when shared genes exist
    if not shared_genes:
        logger.warning("No shared genes between sc and st data — skipping cossim.")
    else:
        adata_sc_shared = adata_sc[:, shared_genes]
        adata_st_shared = adata_st[:, shared_genes]

        expr_sums_raw, sizes = compute_leiden_expression_sums(
            adata_sc_shared, leiden_idx, n_leiden
        )

        adata_sc_norm = adata_sc_shared.copy()
        sc.pp.normalize_total(adata_sc_norm, target_sum=1e4)
        sc.pp.log1p(adata_sc_norm)
        expr_sums_norm, _ = compute_leiden_expression_sums(
            adata_sc_norm, leiden_idx, n_leiden
        )

        # Substate merge coherence — on the normalized+log1p shared-gene
        # Leiden-cluster centroids (mean expression per cluster).
        logger.info("Computing substate merge coherence...")
        centroids_leiden_norm = expr_sums_norm / (sizes[:, None] + 1e-8)
        coherence = compute_substate_coherence(centroids_leiden_norm, G_hard)

        adata_st_norm = adata_st_shared.copy()
        sc.pp.normalize_total(adata_st_norm, target_sum=1e4)
        sc.pp.log1p(adata_st_norm)

        centroids_soft_raw = assemble_state_centroids(G, expr_sums_raw, sizes)
        centroids_hard_raw = assemble_state_centroids(G_hard, expr_sums_raw, sizes)
        centroids_soft_norm = assemble_state_centroids(G, expr_sums_norm, sizes)
        centroids_hard_norm = assemble_state_centroids(G_hard, expr_sums_norm, sizes)

        combos = {
            "soft-raw": (P, centroids_soft_raw, adata_st_shared),
            "hard-raw": (P_hard, centroids_hard_raw, adata_st_shared),
            "soft-norm": (P, centroids_soft_norm, adata_st_norm),
            "hard-norm": (P_hard, centroids_hard_norm, adata_st_norm),
        }
        cossim_dir = data_dir / "cossim"
        cossim_results: dict[str, CossimResult] = {}
        for label, (m, centroids, st_ref) in combos.items():
            pred = predict_expression(m, centroids)  # S x G_shared
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

    # ── 4. Existing UMAP/modularity/contingency/state-profile pipeline ──────
    cell_state_soft = G[leiden_idx].copy()  # (n_cells x L) cell -> state
    cell_state_soft[cell_state_soft < 0.1] = 0.0
    spot_state_soft = P.copy()  # (S x L) spot -> state
    spot_state_soft[spot_state_soft < 0.1] = 0.0

    results = run_analysis(
        adata_sc=adata_sc,
        adata_st=adata_st,
        cell_state_soft=cell_state_soft,
        spot_state_soft=spot_state_soft,
        output_dir=analysis_dir,
        leiden_resolution=leiden_resolution,
        n_leiden=n_leiden,
        leiden_labels=leiden_idx,
    )
    # Persist the biology metrics BEFORE the report is generated — the report's
    # spatial/coherence sections read biology_metrics.json from disk, so it must
    # already exist when generate_analysis_report runs.
    with open(data_dir / "biology_metrics.json", "w") as f:
        json.dump({"spatial": spatial, "coherence": coherence}, f, indent=2)

    generate_analysis_report(analysis_dir, n_active_states, n_mapped_states, n_leiden)
    logger.info("Analysis report written to %s", analysis_dir)

    # ── 5. Flat objective scalars ───────────────────────────────────────────
    objectives = _collect_objectives(
        results=results,
        onehot_P=onehot_P,
        onehot_G=onehot_G,
        cossim_summary=cossim_summary,
        spatial=spatial,
        coherence=coherence,
        n_leiden=n_leiden,
        n_active_states=n_active_states,
        n_mapped_states=n_mapped_states,
    )
    pd.DataFrame([objectives]).to_csv(data_dir / "objective_metrics.csv", index=False)
    logger.info("Objective metrics written to %s", data_dir / "objective_metrics.csv")

    return results, n_leiden, objectives


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Run post-mapping analysis on an existing AIM output folder "
        "(reads mapping_prob.h5ad / leiden_merge_prob.h5ad / leiden_overclustering.h5ad)."
    )
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Folder containing the AIM run's saved mapping outputs; analysis/ is written here",
    )
    args = parser.parse_args()

    analyze_run(
        args.scdata,
        args.stdata,
        args.output_folder,
    )
