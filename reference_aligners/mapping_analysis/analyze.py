"""Analyze a reference aligner's mapping_prob.h5ad output.

Tangram, TACCO, and DOT all write the same spots x cell-type layout with real
cell-type names in var_names, so this single script covers all three — no
aligner-specific logic needed.

Reads mapping_prob.h5ad from mapping_folder plus the original sc/st input
files, and writes plots + machine-readable data + a Typst PDF report to
mapping_folder/analysis/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from metrics.cossim import CossimResult, compute_and_save_cossim
from metrics.onehot import hard_mapping, onehot_metrics

from .metrics import (
    celltype_centroids,
    predict_expression,
    top_marker_genes,
)
from .plots import (
    build_celltype_palette,
    plot_celltype_centroid_zscores,
    plot_cossim_boxplots,
    plot_dominance_thresholds,
    plot_onehot_distribution,
    plot_sc_umap_by_celltype,
    plot_spatial_hard_celltypes,
)
from .report import generate_report

logger = logging.getLogger(__name__)


def analyze_mapping(
    sc_path: Path,
    st_path: Path,
    mapping_folder: Path,
    cell_type_key: str = "cellType",
    top_n_markers: int = 10,
) -> Path | None:
    """
    Run the full mapping analysis and write everything under
    mapping_folder/analysis/.

    Args:
        sc_path, st_path: Full paths to the sc/st h5ad used to produce the mapping.
        mapping_folder: Folder containing mapping_prob.h5ad (as written by
                        run_tangram.py / run_tacco.py / run_dot.py).
        cell_type_key: obs column in sc data with the same values as the
                       mapping's var_names.
        top_n_markers: Marker genes per cell type (union) shown in the
                       centroid z-score heatmap.

    Returns:
        Path to the written report.pdf, or None if Typst compilation failed.
    """
    mapping_folder = Path(mapping_folder)
    mapping_path = mapping_folder / "mapping_prob.h5ad"
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping_prob.h5ad not found in {mapping_folder}")

    analysis_dir = mapping_folder / "analysis"
    plots_dir = analysis_dir / "plots"
    data_dir = analysis_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading mapping and input data...")
    mapping_ad = ad.read_h5ad(mapping_path)  # S x T
    cell_types = mapping_ad.var_names.tolist()
    mapping = np.asarray(mapping_ad.X, dtype=np.float32)
    spot_names = mapping_ad.obs_names.tolist()

    adata_sc = ad.read_h5ad(sc_path)
    adata_st = ad.read_h5ad(st_path)
    if cell_type_key not in adata_sc.obs.columns:
        raise KeyError(
            f"cell_type_key '{cell_type_key}' not found in sc obs columns: "
            f"{list(adata_sc.obs.columns)}"
        )

    # ── 1. Mapping sharpness ("how one-hot") ────────────────────────────────
    logger.info("Computing one-hot metrics...")
    oh_metrics = onehot_metrics(mapping)
    pd.DataFrame(
        {
            "spot": spot_names,
            "max_prob": oh_metrics["max_prob"],
            "gini_impurity": oh_metrics["gini_impurity"],
            "entropy": oh_metrics["entropy"],
        }
    ).to_csv(data_dir / "onehot_metrics_per_spot.csv", index=False)
    with open(data_dir / "onehot_metrics_summary.json", "w") as f:
        json.dump(
            {
                "n_spots": oh_metrics["n_rows"],
                "n_types": oh_metrics["n_cols"],
                **oh_metrics["summary"],
            },
            f,
            indent=2,
        )
    plot_onehot_distribution(oh_metrics, plots_dir / "onehot_distribution.png")
    plot_dominance_thresholds(oh_metrics, plots_dir / "onehot_threshold_fractions.png")

    # ── 2. Hard mapping (argmax) ─────────────────────────────────────────────
    logger.info("Computing hard (argmax) mapping...")
    hard = hard_mapping(mapping)  # S x T one-hot
    hard_labels = np.array([cell_types[i] for i in hard.argmax(axis=1)])
    AnnData(
        X=hard.astype(np.float32),
        obs=pd.DataFrame(index=spot_names),
        var=pd.DataFrame(index=cell_types),
    ).write_h5ad(data_dir / "mapping_hard.h5ad")
    pd.DataFrame({"spot": spot_names, "cell_type": hard_labels}).to_csv(
        data_dir / "spot_hard_celltype.csv", index=False
    )

    # ── 3. SC UMAP by cell type ──────────────────────────────────────────────
    logger.info("Computing SC UMAP...")
    celltype_palette = build_celltype_palette(cell_types)
    plot_sc_umap_by_celltype(
        adata_sc,
        cell_type_key,
        cell_types,
        celltype_palette,
        plots_dir / "sc_umap_celltype.png",
    )
    cell_type_counts = adata_sc.obs[cell_type_key].astype(str).value_counts().to_dict()

    # ── 4. Cell-type centroid z-scores (normalized + log1p, all genes) ──────
    logger.info("Computing cell-type centroids and marker genes...")
    adata_sc_norm = adata_sc.copy()
    sc.pp.normalize_total(adata_sc_norm, target_sum=1e4)
    sc.pp.log1p(adata_sc_norm)

    centroids_norm_full = celltype_centroids(adata_sc_norm, cell_type_key, cell_types)
    AnnData(
        X=centroids_norm_full.to_numpy().astype(np.float32),
        obs=pd.DataFrame(index=centroids_norm_full.index),
        var=pd.DataFrame(index=centroids_norm_full.columns),
    ).write_h5ad(data_dir / "celltype_centroids_normlog_allgenes.h5ad")

    markers = top_marker_genes(adata_sc_norm, cell_type_key, top_n_markers)
    markers = [g for g in markers if g in centroids_norm_full.columns]
    plot_celltype_centroid_zscores(
        centroids_norm_full[markers], plots_dir / "celltype_centroid_zscores.png"
    )

    # ── 5. Spatial hard-mapping plot ─────────────────────────────────────────
    logger.info("Plotting spatial hard-mapped cell types...")
    plot_spatial_hard_celltypes(
        adata_st,
        hard_labels,
        cell_types,
        celltype_palette,
        plots_dir / "spatial_hard_celltypes.png",
    )

    # ── 6. Reconstruction cosine similarity (soft/hard x raw/norm) ───────────
    logger.info("Computing reconstruction cosine similarities...")
    shared_genes = sorted(set(adata_sc.var_names) & set(adata_st.var_names))
    cossim_summary: dict[str, dict] = {}
    if not shared_genes:
        logger.warning("No shared genes between sc and st data — skipping cossim.")
    else:
        centroids_raw = celltype_centroids(
            adata_sc[:, shared_genes], cell_type_key, cell_types
        )
        centroids_norm_shared = centroids_norm_full[shared_genes]

        adata_st_norm = adata_st.copy()
        sc.pp.normalize_total(adata_st_norm, target_sum=1e4)
        sc.pp.log1p(adata_st_norm)

        combos = {
            "soft-raw": (mapping, centroids_raw, adata_st),
            "hard-raw": (hard, centroids_raw, adata_st),
            "soft-norm": (mapping, centroids_norm_shared, adata_st_norm),
            "hard-norm": (hard, centroids_norm_shared, adata_st_norm),
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
                st_ref[:, shared_genes], pred_adata, cossim_dir, suffix=f"-{label}"
            )
            cossim_results[label] = result
            cossim_summary[label] = {
                "median_gene": result.median_gene,
                "median_spot": result.median_spot,
            }
        pd.DataFrame(cossim_summary).T.to_csv(data_dir / "cossim_summary.csv")
        plot_cossim_boxplots(cossim_results, plots_dir / "cossim_boxplots.png")

    # ── Report ───────────────────────────────────────────────────────────────
    logger.info("Generating report...")
    report_path = generate_report(
        analysis_dir,
        mapping_source=str(mapping_folder),
        onehot_summary=oh_metrics,
        cell_type_counts=cell_type_counts,
        cossim_summary=cossim_summary,
    )
    if report_path:
        logger.info("Report written to %s", report_path)
    return report_path
