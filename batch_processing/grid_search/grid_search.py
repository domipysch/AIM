import argparse
import copy
import csv
import itertools
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
import anndata as ad
import pandas as pd
import scanpy as sc
import yaml
import main as aim_main
from evaluate_k.analysis import run_analysis
from evaluate_k.clustering import run_leiden_shared_genes
from evaluate_k.report import (
    generate_per_k_report,
    generate_summary_report,
)
from utils import run_pca_neighbors_umap
from metrics import (
    run_all_metrics,
    run_all_shared_boxplots,
    run_all_permutation_boxplots,
)

logger = logging.getLogger(__name__)


def _read_median_cossim(run_dir: Path) -> tuple[float | None, float | None]:
    """Return (gene_median, spot_median) from the metrics cossim JSONs, or (None, None)."""
    metrics_dir = run_dir / "metrics"

    def _read_median(json_path: Path) -> float | None:
        if not json_path.exists():
            return None
        with open(json_path) as f:
            return float(json.load(f)["median"])

    gene_median = _read_median(metrics_dir / "cossim-per-gene.json")
    spot_median = _read_median(metrics_dir / "cossim-per-spot.json")
    return gene_median, spot_median


def _precompute_analysis_artifacts(
    sc_path: Path, st_path: Path, leiden_resolution: float
):
    """Pre-compute the per-dataset analysis artifacts shared across all K runs.

    Returns (adata_sc, adata_st, adata_processed_base, leiden_labels_precomp,
    leiden_shared_labels_precomp, shared_genes).
    """
    logger.info("Pre-computing analysis artifacts (PCA, UMAP, Leiden)…")
    adata_sc = ad.read_h5ad(sc_path)
    adata_st = ad.read_h5ad(st_path)
    shared_genes = list(set(adata_sc.var_names) & set(adata_st.var_names))

    adata_processed_base = adata_sc.copy()
    run_pca_neighbors_umap(adata_processed_base)
    sc.tl.leiden(
        adata_processed_base, resolution=leiden_resolution, key_added="_leiden_ref"
    )
    leiden_labels_precomp = adata_processed_base.obs["_leiden_ref"].astype(int).values

    leiden_shared_labels_precomp, _ = run_leiden_shared_genes(
        adata_sc, shared_genes=shared_genes, resolution=leiden_resolution
    )
    logger.info("Analysis pre-computation done.")
    return (
        adata_sc,
        adata_st,
        adata_processed_base,
        leiden_labels_precomp,
        leiden_shared_labels_precomp,
        shared_genes,
    )


def _analyze_run(
    run_dir: Path,
    run_id,
    adata_sc,
    adata_st,
    leiden_resolution: float,
    adata_processed_base,
    leiden_labels_precomp,
    leiden_shared_labels_precomp,
) -> dict | None:
    """Run post-mapping analysis + per-K report for one existing run.

    Reads the stored cell->state matrix B (intermediate/B_thresh.h5ad) and the
    spot->state matrix C from mapping_prob.h5ad (transposed; run_analysis only
    argmaxes C). No mapping recompute. K is always derived from B.shape[1].
    Returns the analysis-overview row dict, or None when the files are missing.
    """
    b_path = run_dir / "intermediate" / "B_thresh.h5ad"
    mapping_path = run_dir / "mapping_prob.h5ad"
    if not (b_path.exists() and mapping_path.exists()):
        logger.warning(
            "Run %s: B_thresh / mapping_prob files missing — skipping analysis",
            run_id,
        )
        return None

    B = ad.read_h5ad(b_path).X
    # mapping_prob is (states x spots) = H.T; transpose back to (spots x states)
    C = ad.read_h5ad(mapping_path).X.T
    K = int(B.shape[1])
    analysis_dir = run_dir / "analysis"
    results = run_analysis(
        adata_sc=adata_sc,
        adata_st=adata_st,
        B=B,
        C=C,
        output_dir=analysis_dir,
        K=K,
        leiden_resolution=leiden_resolution,
        adata_processed_base=adata_processed_base,
        leiden_labels=leiden_labels_precomp,
        leiden_shared_labels=leiden_shared_labels_precomp,
    )
    generate_per_k_report(analysis_dir, K, str(run_id))

    gene_median, spot_median = _read_median_cossim(run_dir)
    analysis_row: dict = {"run": str(run_id), "K": K}
    analysis_row["Computed states"] = results["n_computed_states"]
    analysis_row["Computed states > 1%"] = results["n_computed_states_above_1pct"]
    analysis_row["Mapped states"] = results["n_mapped_states"]
    analysis_row["Mapped states > 1%"] = results["n_mapped_states_above_1pct"]
    analysis_row["per_state_perm_p"] = results["substate_metrics"]["weighted_perm_p"]
    for metric, value in results["metrics_computed"].items():
        analysis_row[f"{metric}__computed"] = value
    analysis_row["contingency_score"] = results["contingency_matching"]["score"]
    analysis_row["median_cossim_gene"] = gene_median
    analysis_row["median_cossim_spot"] = spot_median
    return analysis_row


def _write_summary_analysis(ds_folder: Path, analysis_summary_rows: list[dict]) -> None:
    """Write analysis_overview.csv and the cross-K summary report PDF."""
    if not analysis_summary_rows:
        logger.warning("No analysis rows — skipping summary analysis report.")
        return
    overview_path = ds_folder / "analysis_overview.csv"
    df = pd.DataFrame(analysis_summary_rows).set_index("run").T
    df.to_csv(overview_path, index=True)
    logger.info("Analysis overview → %s", overview_path)
    generate_summary_report(ds_folder)


def run_config(
    sc_path: Path,
    st_path: Path,
    run_config_path: Path,
    output_folder: Path,
    metrics_folder: Path,
    gpu_limit_gb: int = 6,
) -> dict:
    # Load config from the per-run YAML written by the grid-search loop
    with open(run_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    loss_cfg = cfg.get("loss_weights", {})

    verbose_flag = logger.getEffectiveLevel() == logging.DEBUG

    # Run alignment (G x S)
    predicted_gep, predicted_gep_det, cell_to_celltype, losses_after_last_epoch = (
        aim_main.main(
            sc_path,
            st_path,
            output_folder=output_folder,
            lr=cfg["lr"],
            epochs=cfg["epochs"],
            normalize_and_log=cfg.get("normalize_and_log", False),
            leiden_resolution=cfg.get("reference_leiden_clustering_resolution", 3.0),
            lambda_rec_spot=loss_cfg.get("lambda_rec_spot", 0.5),
            lambda_rec_gene=loss_cfg.get("lambda_rec_gene", 0.5),
            # lambda_clust_intra=loss_cfg.get("lambda_clust_intra", 0.0),
            # lambda_clust_inter=loss_cfg.get("lambda_clust_inter", 0.0),
            lambda_state_entropy=loss_cfg.get("lambda_state_entropy", 0.1),
            lambda_spot_entropy=loss_cfg.get("lambda_spot_entropy", 0.08),
            lambda_merge_entropy=loss_cfg.get("lambda_merge_entropy", 1.0),
            lambda_merge_coherence=loss_cfg.get("lambda_merge_coherence", 0.5),
            verbose_logging=verbose_flag,
            store_intermediate=True,
            skip_analysis=True,
            gpu_limit_gb=gpu_limit_gb,
        )
    )

    # Run individual metrics (probabilistic)
    run_all_metrics.main(
        sc_path,
        st_path,
        metrics_folder,
        result_gep=predicted_gep,
    )

    # Run individual metrics (deterministic)
    run_all_metrics.main(
        sc_path,
        st_path,
        metrics_folder,
        result_gep=predicted_gep_det,
        name_suffix="-det",
    )

    return losses_after_last_epoch


def main(
    experiment_config: Path,
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
    gpu_limit_gb: int = 6,
):

    if not experiment_config.exists():
        raise FileNotFoundError(f"experiment_config not found: {experiment_config}")

    # Load experiment config
    with open(experiment_config, "r") as f:
        base_cfg = yaml.safe_load(f) or {}
    if not isinstance(base_cfg, dict):
        raise ValueError("Top-level experiment_config must be a mapping/dict.")

    # Remove data/output sections if present in the YAML (ignored — CLI args take precedence)
    base_cfg.pop("data", None)
    base_cfg.pop("output", None)

    if not sc_path.exists():
        raise FileNotFoundError(f"sc.h5ad not found: {sc_path}")
    if not st_path.exists():
        raise FileNotFoundError(f"st.h5ad not found: {st_path}")

    # Helper: collect leaf paths -> list of values (lists become value lists, scalars become singleton list)
    def collect_leaves(node, path=()):
        """
        Traverses `node` (which may be nested dicts/lists/scalars) and returns a list of
        (path_tuple, values_list) pairs.

        New behavior: if at some path the node is a list AND every element of that list is a dict,
        we treat the whole list as a set of alternative full-config dictionaries for that path.
        This allows specifying e.g. multiple complete `loss_weights` dicts as alternative configs.

        Examples handled:
        - scalar -> becomes [scalar]
        - list of scalars -> becomes that list
        - list of dicts -> treated as a leaf; values_list equals the list of dicts
        - dict -> recurse into keys
        """
        leaves = []
        # If it's a list at this path
        if isinstance(node, list):
            # empty list -> keep as-is (will be validated later)
            if len(node) == 0:
                leaves.append((path, []))
                return leaves
            # if all items are dicts, treat the whole list as an atomic set of dict-options
            if all(isinstance(item, dict) for item in node):
                leaves.append((path, node))
                return leaves
            # otherwise treat it as a normal list of scalar options
            leaves.append((path, node))
            return leaves

        # If it's a dict, recurse into keys
        if isinstance(node, dict):
            for k, v in node.items():
                leaves.extend(collect_leaves(v, path + (k,)))
            return leaves

        # Otherwise scalar leaf
        vals = [node]
        leaves.append((path, vals))
        return leaves

    leaves = collect_leaves(base_cfg)

    # Ensure no leaf has empty list
    for path, vals in leaves:
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(
                f"Configuration entry {'.'.join(path)} must be a non-empty list or scalar."
            )

    # Prepare ordered lists for product
    paths = [p for p, v in leaves]
    lists = [v for p, v in leaves]

    total_runs = 1
    for v in lists:
        total_runs *= len(v)

    logger.info(
        f"Experiment config loaded from {experiment_config}. "
        f"Total runs: {total_runs}"
    )

    # Function to set a value in nested dict by path
    def set_in_dict(d: dict, path: tuple, value):
        cur = d
        for key in path[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[path[-1]] = value

    ds_folder = output_folder

    logger.info(f"=== SC: {sc_path.stem}  ST: {st_path.stem} ===")

    ds_folder.mkdir(parents=True, exist_ok=True)

    # Copy experiment config to output folder for reference
    shutil.copy(experiment_config, ds_folder / "experiment_config.yml")

    # ── Analysis artifacts (PCA/UMAP/Leiden) are precomputed lazily per Leiden
    # resolution and cached. The resolution can be a grid axis, so a single
    # up-front precompute would be wrong for runs at other resolutions. ──
    analysis_summary_rows: list[dict] = []
    _analysis_cache: dict[float, tuple] = {}

    def _get_analysis_artifacts(resolution: float):
        if resolution not in _analysis_cache:
            _analysis_cache[resolution] = _precompute_analysis_artifacts(
                sc_path, st_path, resolution
            )
        return _analysis_cache[resolution]

    # Prepare summary CSV
    summary_path = ds_folder / "summary.csv"

    # Iterate over grid
    combo_iter = itertools.product(*lists)

    run_id = 0
    _fixed_cols = [
        "id",
        "config_path",
        "output_folder",
        "status",
        "duration_seconds",
        "error_message",
    ]
    with open(summary_path, "w", newline="") as summary_file:
        # DictWriter is created after the first run, once loss column names are known.
        writer = None

        for combo in combo_iter:
            # Build run-specific config
            cfg_copy = copy.deepcopy(base_cfg)
            for path, val in zip(paths, combo):
                set_in_dict(cfg_copy, path, val)

            run_dir = ds_folder / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            run_config_path = run_dir / "config.yml"
            with open(run_config_path, "w") as cf:
                yaml.safe_dump(cfg_copy, cf, sort_keys=False)

            metric_dir = run_dir / "metrics"
            metric_dir.mkdir(parents=True, exist_ok=True)

            start = time.time()
            exc = None
            tb = ""
            losses_after_last_epoch = {}
            status = "error"
            error_msg = ""

            try:
                logger.info(
                    f"Starting run {run_id}/{total_runs - 1} -> writing to {run_dir}"
                )
                losses_after_last_epoch = run_config(
                    sc_path,
                    st_path,
                    run_config_path,
                    run_dir,
                    metric_dir,
                    gpu_limit_gb=gpu_limit_gb,
                )
                status = "ok"
            except Exception as e:
                exc = e
                error_msg = str(e)
                tb = traceback.format_exc()

            duration = time.time() - start
            if exc is None:
                logger.info(f"Run {run_id} completed in {duration:.2f}s")
            else:
                logger.error(f"Run {run_id} failed after {duration:.2f}s: {exc}\n{tb}")

            # ── Per-run analysis & report ─────────────────────────────────
            if exc is None:
                try:
                    run_res = float(
                        cfg_copy.get("reference_leiden_clustering_resolution", 3.0)
                    )
                    (
                        adata_sc,
                        adata_st,
                        adata_processed_base,
                        leiden_labels_precomp,
                        leiden_shared_labels_precomp,
                        shared_genes,
                    ) = _get_analysis_artifacts(run_res)
                    analysis_row = _analyze_run(
                        run_dir,
                        run_id,
                        adata_sc,
                        adata_st,
                        run_res,
                        adata_processed_base,
                        leiden_labels_precomp,
                        leiden_shared_labels_precomp,
                    )
                    if analysis_row is not None:
                        analysis_summary_rows.append(analysis_row)
                except Exception as _analysis_exc:
                    logger.error(
                        "Analysis failed for run %s: %s", run_id, _analysis_exc
                    )

            row = {
                "id": run_id,
                "config_path": str(run_config_path),
                "output_folder": str(run_dir),
                "status": status,
                "duration_seconds": f"{duration:.3f}",
                "error_message": error_msg,
                **losses_after_last_epoch,
            }
            if writer is None:
                fieldnames = _fixed_cols + [k for k in losses_after_last_epoch]
                writer = csv.DictWriter(
                    summary_file, fieldnames=fieldnames, extrasaction="ignore"
                )
                writer.writeheader()
            writer.writerow(row)
            summary_file.flush()

            if exc is not None:
                raise exc

            run_id += 1

    # Create shared boxplots for this dataset (probabilistic metrics, one box per run)
    metric_dirs = [ds_folder / str(i) / "metrics" for i in range(run_id)]
    labels = [str(i) for i in range(run_id)]
    shared_dir = ds_folder / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    # Run shared metrics
    run_all_shared_boxplots.main(
        metric_dirs,
        labels,
        shared_dir,
    )
    # Run shared permutation test boxplots
    # run_all_permutation_boxplots.main(
    #     metric_dirs,
    #     labels,
    #     shared_dir,
    # )

    # ── Summary analysis report ───────────────────────────────────────────
    _write_summary_analysis(ds_folder, analysis_summary_rows)


def run_analyses_only(
    experiment_config: Path,
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
) -> None:
    """Re-run only the post-mapping analyses for an already-computed dataset.

    Reuses each run's stored B/C assignment matrices (``<run>/intermediate/``) —
    no mapping/training is recomputed. For every numeric run sub-folder it reruns
    ``run_analysis`` + the per-K report, then rewrites ``analysis_overview.csv``
    and the summary report. Mapping outputs, metrics and shared boxplots are left
    untouched. The per-run ``K`` is read from each ``<run>/config.yml``; the Leiden
    resolution is read from ``experiment_config`` (matching the original run).
    """
    experiment_config = Path(experiment_config)
    ds_folder = Path(output_folder)
    if not ds_folder.is_dir():
        raise FileNotFoundError(
            f"Output folder not found (run the mapping first): {ds_folder}"
        )
    if not experiment_config.exists():
        raise FileNotFoundError(f"experiment_config not found: {experiment_config}")

    with open(experiment_config, "r") as f:
        base_cfg = yaml.safe_load(f) or {}
    leiden_resolution = float(
        base_cfg.get("reference_leiden_clustering_resolution", 3.0)
    )

    logger.info(f"=== Analyses-only: SC: {sc_path.stem}  ST: {st_path.stem} ===")

    (
        adata_sc,
        adata_st,
        adata_processed_base,
        leiden_labels_precomp,
        leiden_shared_labels_precomp,
        shared_genes,
    ) = _precompute_analysis_artifacts(sc_path, st_path, leiden_resolution)

    run_dirs = sorted(
        (d for d in ds_folder.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not run_dirs:
        logger.warning(
            "No run sub-folders found in %s — nothing to analyse.", ds_folder
        )
        return

    analysis_summary_rows: list[dict] = []
    for run_dir in run_dirs:
        run_id = int(run_dir.name)
        run_config_path = run_dir / "config.yml"
        if not run_config_path.exists():
            logger.warning("Run %s: config.yml missing — skipping.", run_id)
            continue
        try:
            analysis_row = _analyze_run(
                run_dir,
                run_id,
                adata_sc,
                adata_st,
                leiden_resolution,
                adata_processed_base,
                leiden_labels_precomp,
                leiden_shared_labels_precomp,
            )
            if analysis_row is not None:
                analysis_summary_rows.append(analysis_row)
                logger.info("Run %s analysed.", run_id)
        except Exception as _analysis_exc:
            logger.error("Analysis failed for run %s: %s", run_id, _analysis_exc)

    _write_summary_analysis(ds_folder, analysis_summary_rows)


if __name__ == "__main__":

    # 1. Parse Arguments
    parser = argparse.ArgumentParser(
        description="Run AIM alignment. Hyperparameters come from the experiment YAML; data paths are passed as arguments."
    )
    parser.add_argument(
        "--scdata",
        type=Path,
        required=True,
        help="Path to the scRNA-seq .h5ad file.",
    )
    parser.add_argument(
        "--stdata",
        type=Path,
        required=True,
        help="Path to the spatial transcriptomics .h5ad file.",
    )
    parser.add_argument(
        "-c",
        "--experiment_config",
        type=Path,
        required=True,
        help="Path(s) to experiment config YAML. Multiple configs are run sequentially.",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Output folder for results.",
    )
    parser.add_argument(
        "--logging",
        dest="logging",
        choices=["normal", "verbose"],
        default="normal",
        help="Logging verbosity. Use 'verbose' for more logs.",
    )
    parser.add_argument(
        "--gpu_limit_gb",
        type=int,
        default=48,
        help="GPU memory limit in GB. Abort if estimated usage exceeds this value (default: 6).",
    )
    args = parser.parse_args()

    # 2. Configure logging based on argument
    level = logging.DEBUG if args.logging == "verbose" else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(level)

    # 3. Run
    main(
        args.experiment_config,
        sc_path=args.scdata,
        st_path=args.stdata,
        output_folder=args.output_folder,
        gpu_limit_gb=args.gpu_limit_gb,
    )
