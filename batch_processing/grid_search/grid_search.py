import argparse
import copy
import csv
import itertools
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
import pandas as pd
import yaml
import main as aim_main

logger = logging.getLogger(__name__)


def _read_losses_end(output_folder: Path) -> dict:
    """Read the final-epoch loss values that main() writes to loss/losses_end.csv."""
    losses_path = output_folder / "loss" / "losses_end.csv"
    if not losses_path.exists():
        return {}
    df = pd.read_csv(losses_path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def run_config(
    sc_path: Path,
    st_path: Path,
    run_config_path: Path,
    output_folder: Path,
    gpu_limit_gb: int = 6,
) -> dict:
    # Load config from the per-run YAML written by the grid-search loop
    with open(run_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    loss_cfg = cfg.get("loss_weights", {})

    verbose_flag = logger.getEffectiveLevel() == logging.DEBUG

    # Run alignment — writes mapping_prob.h5ad, leiden_merge_prob.h5ad,
    # clusters_prob.h5ad, and loss/ to output_folder.
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
        lambda_state_entropy=loss_cfg.get("lambda_state_entropy", 0.1),
        lambda_spot_entropy=loss_cfg.get("lambda_spot_entropy", 0.0),
        lambda_spot_gini=loss_cfg.get("lambda_spot_gini", 0.5),
        lambda_merge_entropy=loss_cfg.get("lambda_merge_entropy", 0.0),
        lambda_merge_gini=loss_cfg.get("lambda_merge_gini", 1.0),
        lambda_merge_coherence=loss_cfg.get("lambda_merge_coherence", 0.5),
        verbose_logging=verbose_flag,
        gpu_limit_gb=gpu_limit_gb,
    )

    return _read_losses_end(output_folder)


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
