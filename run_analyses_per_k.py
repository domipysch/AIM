"""
Run post-mapping analysis for every K-run inside an experiment output folder.

Usage
-----
    python -m run_analyses_per_k -r <output_folder> [<output_folder2> ...]

Example
-------
    python -m run_analyses_per_k \
        -r C:/Users/zi69hebi/Dev/10_Alignment/Data/05_Experiments/04_ColorectalCancer

Expected output_folder layout (as produced by run_experiment.py)
----------------------------------------------------------------
    <output_folder>/
      <sc_stem>__<st_stem>/
        experiment_config.yml   ← contains sc_paths / st_paths
        summary.csv
        0/
          config.yml            ← contains model.K
          intermediate/
            B_thresh.h5ad       ← soft cell-to-state matrix  (n_cells × K)
            C_thresh.h5ad       ← soft spot-to-state matrix  (n_spots × K)
        1/ ...
        metrics/
          0/                    ← per-run metrics (o2/, o4/, ...)
          1/ ...

Outputs (per run, written next to intermediate/ and loss/)
----------------------------------------------------------
    <run_folder>/analysis/
        cell_mapping.csv
        spot_mapping.csv
        cell_state_fractions.png
        cell_state_profiles.h5ad
        umap_computed.png
        umap_leiden.png
        metrics_comparison.csv
        centroid_matching_scores.csv

Overview (written once after all runs)
---------------------------------------
    <sc_stem>__<st_stem>/analysis_overview.csv
        One row per run: K, all clustering metrics (computed / leiden / leiden_shared),
        Hungarian cosine similarity, greedy cosine similarity.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import yaml

from src.alternative_idea.src.evaluate_k.analysis import run_analysis
from src.alternative_idea.src.evaluate_k.clustering import run_leiden_shared_genes
from src.alternative_idea.src.evaluate_k.report import (
    generate_per_k_report,
    generate_summary_report,
)
from src.alternative_idea.src.utils import run_pca_neighbors_umap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)


def _find_pair_dir(result_folder: Path) -> Path:
    """Return the single <sc>__<st> subdirectory."""
    candidates = [d for d in result_folder.iterdir() if d.is_dir()]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one sc__st subdirectory in {result_folder}, "
            f"found: {[d.name for d in candidates]}"
        )
    return candidates[0]


def _load_experiment_config(pair_dir: Path) -> dict:
    config_path = pair_dir / "experiment_config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"experiment_config.yml not found in {pair_dir}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _run_dirs(pair_dir: Path) -> list[Path]:
    """Return all numbered run directories, sorted numerically."""
    dirs = [d for d in pair_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(dirs, key=lambda d: int(d.name))


def _load_K(run_dir: Path) -> int:
    config_path = run_dir / "config.yml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return int(cfg["model"]["K"])


def _read_median_cossim(
    metrics_folder: Path, run_id: str
) -> tuple[float | None, float | None]:
    """Return (gene_median, spot_median) from the o2 cossim JSONs, or (None, None)."""
    run_subdir = metrics_folder / run_id
    if not run_subdir.is_dir():
        logger.warning("metrics run dir not found: %s", run_subdir)
        return None, None

    def _read_median(json_path: Path) -> float | None:
        if not json_path.exists():
            logger.warning("cossim JSON not found: %s", json_path)
            return None
        with open(json_path) as f:
            return float(json.load(f)["median"])

    gene_median = _read_median(run_subdir / "o2" / "boxplots_per_gene" / "cossim.json")
    spot_median = _read_median(run_subdir / "o2" / "boxplots_per_spot" / "cossim.json")
    return gene_median, spot_median


def main(result_folder: Path) -> None:
    result_folder = Path(result_folder)

    pair_dir = _find_pair_dir(result_folder)
    logger.info("Pair directory: %s", pair_dir)

    metrics_folder = pair_dir / "metrics"

    exp_cfg = _load_experiment_config(pair_dir)
    sc_path = Path(exp_cfg["data"]["sc_paths"][0])
    st_path = Path(exp_cfg["data"]["st_paths"][0])
    leiden_resolution: float = float(
        exp_cfg.get("training", {}).get("reference_leiden_clustering_resolution")
    )

    logger.info("Loading sc data from %s", sc_path)
    adata_sc = ad.read_h5ad(sc_path)
    logger.info("Loading st data from %s", st_path)
    adata_st = ad.read_h5ad(st_path)

    # ── Pre-compute K-independent artifacts (shared across all runs) ──────────
    shared_genes = list(set(adata_sc.var_names) & set(adata_st.var_names))

    logger.info("Pre-computing adata_processed (normalize → PCA → neighbors → UMAP)…")
    adata_processed_base = adata_sc.copy()
    run_pca_neighbors_umap(adata_processed_base)

    logger.info(
        "Pre-computing Leiden clustering (all genes, resolution=%.2f)…",
        leiden_resolution,
    )
    sc.tl.leiden(
        adata_processed_base, resolution=leiden_resolution, key_added="_leiden_ref"
    )
    leiden_labels_precomp = adata_processed_base.obs["_leiden_ref"].astype(int).values
    logger.info("Leiden (all genes): %d clusters", len(set(leiden_labels_precomp)))

    logger.info(
        "Pre-computing Leiden clustering (shared genes, resolution=%.2f)…",
        leiden_resolution,
    )
    leiden_shared_labels_precomp, adata_shared_precomp = run_leiden_shared_genes(
        adata_sc, shared_genes=shared_genes, resolution=leiden_resolution
    )
    logger.info(
        "Leiden (shared genes): %d clusters", len(set(leiden_shared_labels_precomp))
    )

    run_dirs = _run_dirs(pair_dir)
    if not run_dirs:
        logger.error("No numbered run directories found in %s", pair_dir)
        sys.exit(1)

    logger.info("Found %d run(s): %s", len(run_dirs), [d.name for d in run_dirs])

    summary_rows: list[dict] = []

    for run_dir in run_dirs:
        K = _load_K(run_dir)
        intermediate = run_dir / "intermediate"

        b_path = intermediate / "B_thresh.h5ad"
        c_path = intermediate / "C_thresh.h5ad"

        if not b_path.exists() or not c_path.exists():
            logger.warning(
                "Run %s: B_thresh.h5ad or C_thresh.h5ad missing — skipping",
                run_dir.name,
            )
            continue

        logger.info("=== Run %s  (K=%d) ===", run_dir.name, K)

        B = ad.read_h5ad(b_path).X
        C = ad.read_h5ad(c_path).X

        output_dir = run_dir / "analysis"
        results = run_analysis(
            adata_sc=adata_sc,
            adata_st=adata_st,
            B=B,
            C=C,
            output_dir=output_dir,
            K=K,
            leiden_resolution=leiden_resolution,
            adata_processed_base=adata_processed_base,
            leiden_labels=leiden_labels_precomp,
            leiden_shared_labels=leiden_shared_labels_precomp,
            adata_shared=adata_shared_precomp,
        )
        logger.info("Run %s done → %s", run_dir.name, output_dir)

        gene_median, spot_median = (
            _read_median_cossim(metrics_folder, run_dir.name)
            if metrics_folder is not None
            else (None, None)
        )
        generate_per_k_report(
            output_dir,
            K,
            run_dir.name,
            median_cossim_gene=gene_median,
            median_cossim_spot=spot_median,
        )

        row: dict = {"run": run_dir.name, "K": K}
        row["Computed states"] = results["n_computed_states"]
        row["Computed states > 1%"] = results["n_computed_states_above_1pct"]
        row["Mapped states"] = results["n_mapped_states"]
        row["per_state_perm_p"] = results["substate_metrics"]["weighted_perm_p"]
        for metric, value in results["metrics_computed"].items():
            row[f"{metric}__computed"] = value
        row["hungarian_cosim"] = results["centroid_matching"]["hungarian_score"]
        row["greedy_cosim"] = results["centroid_matching"]["greedy_score"]
        row["contingency_score"] = results["contingency_matching"]["score"]
        row["median_cossim_gene"] = gene_median
        row["median_cossim_spot"] = spot_median
        summary_rows.append(row)

    if summary_rows:
        overview_path = pair_dir / "analysis_overview.csv"
        df = pd.DataFrame(summary_rows).set_index("run").T
        df.to_csv(overview_path, index=True)
        logger.info("Overview CSV → %s", overview_path)
        generate_summary_report(pair_dir)

    logger.info("All runs complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run post-mapping analysis for every K-run in a result folder."
    )
    parser.add_argument(
        "-r",
        "--result_folder",
        type=Path,
        nargs="+",
        required=True,
        help="output_folder(s) produced by run_experiment.py. Multiple folders are processed sequentially.",
    )
    args = parser.parse_args()

    for result_folder in args.result_folder:
        main(result_folder=result_folder)
