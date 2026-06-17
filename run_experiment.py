import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
import yaml
import itertools
import copy
import time
import csv
import traceback
import logging
from src.alternative_idea import main as alternative_idea_main
from src.metrics import (
    run_all_metrics,
    run_all_shared_boxplots,
    run_all_permutation_boxplots,
)

logger = logging.getLogger(__name__)


def create_shared_boxplots(
    ids: list[str],
    metrics_folder: Path,
    output_folder: Path,
    run_permutation_tests: bool = False,
):

    # Run shared metrics
    run_all_shared_boxplots.main(
        [metrics_folder / s_id for s_id in ids],
        ids,
        output_folder,
    )

    # Run shared permutation test boxplots
    if run_permutation_tests:
        run_all_permutation_boxplots.main(
            [metrics_folder / s_id for s_id in ids],
            ids,
            output_folder,
        )


def run_config(
    sc_path: Path,
    st_path: Path,
    run_config_path: Path,
    save_result_path: Optional[Path],
    metrics_folder: Path,
    metrics_folder_det: Optional[Path],
    run_permutation_tests: bool = False,
    no_gpu_limit: bool = False,
    force_cpu: bool = False,
) -> dict:
    # Determine verbose flag from current logger level
    verbose_flag = logger.getEffectiveLevel() == logging.DEBUG

    # Run alignment (G x S)
    predicted_gep, predicted_gep_det, cell_to_celltype, losses_after_last_epoch = (
        alternative_idea_main.main(
            sc_path,
            st_path,
            run_config_path,
            output_path=save_result_path,
            verbose_logging=verbose_flag,
            store_intermediate=True,
            no_gpu_limit=no_gpu_limit,
            force_cpu=force_cpu,
        )
    )

    # Run individual metrics (probabilistic)
    run_all_metrics.main(
        sc_path,
        st_path,
        metrics_folder,
        result_gep=predicted_gep,
        run_permutation_tests=run_permutation_tests,
    )

    # Run individual metrics (deterministic) if applicable
    if predicted_gep_det is not None:
        assert metrics_folder_det is not None
        run_all_metrics.main(
            sc_path,
            st_path,
            metrics_folder_det,
            result_gep=predicted_gep_det,
            run_permutation_tests=run_permutation_tests,
        )

    return losses_after_last_epoch


def main(
    experiment_config: Path,
    save_result: bool = False,
    run_permutation_tests: bool = False,
    no_gpu_limit: bool = False,
    force_cpu: bool = False,
):

    if not experiment_config.exists():
        raise FileNotFoundError(f"experiment_config not found: {experiment_config}")

    # Load experiment config
    with open(experiment_config, "r") as f:
        base_cfg = yaml.safe_load(f) or {}
    if not isinstance(base_cfg, dict):
        raise ValueError("Top-level experiment_config must be a mapping/dict.")

    # Extract and validate data/output sections (not part of the grid search)
    if "data" not in base_cfg:
        raise ValueError("experiment_config must contain a 'data' section.")
    if "output" not in base_cfg:
        raise ValueError("experiment_config must contain an 'output' section.")
    data_cfg = base_cfg.pop("data")
    output_cfg = base_cfg.pop("output")

    sc_paths = [Path(p) for p in data_cfg["sc_paths"]]
    st_paths = [Path(p) for p in data_cfg["st_paths"]]
    output_folder = Path(output_cfg["output_folder"])

    for sc_path in sc_paths:
        if not sc_path.exists():
            raise FileNotFoundError(f"sc.h5ad not found: {sc_path}")
    for st_path in st_paths:
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

    n_configs = len(sc_paths) * len(st_paths)
    logger.info(
        f"Experiment config loaded from {experiment_config}. "
        f"Total runs per SC×ST config: {total_runs}, "
        f"configs: {n_configs} ({len(sc_paths)} SC × {len(st_paths)} ST)"
    )

    # Function to set a value in nested dict by path
    def set_in_dict(d: dict, path: tuple, value):
        cur = d
        for key in path[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[path[-1]] = value

    for sc_path, st_path in itertools.product(sc_paths, st_paths):
        dataset_name = f"{sc_path.parent.name}_{sc_path.stem}__{st_path.parent.name}_{st_path.stem}"
        ds_folder = output_folder / dataset_name
        ds_metric_folder = ds_folder / "metrics"

        logger.info(f"=== Dataset: {dataset_name} ===")

        if os.path.exists(ds_folder):
            shutil.rmtree(ds_folder)
        ds_folder.mkdir(parents=True, exist_ok=False)
        ds_metric_folder.mkdir(parents=True, exist_ok=False)

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
            "output_path",
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

                result_path = run_dir / "result_GEP.h5ad"

                metric_dir = ds_metric_folder / str(run_id)
                metric_dir.mkdir(parents=True, exist_ok=False)

                # if mode is 'deterministic', then also create a folder "<run_id>_det"
                metric_dir_det = None
                if cfg_copy["mapping"]["deterministic"]:
                    metric_dir_det = ds_metric_folder / f"{run_id}_det"
                    metric_dir_det.mkdir(parents=True, exist_ok=False)

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
                        (run_dir / "gep.h5ad") if save_result else None,
                        metric_dir,
                        metric_dir_det,
                        run_permutation_tests=run_permutation_tests,
                        no_gpu_limit=no_gpu_limit,
                        force_cpu=force_cpu,
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
                    logger.error(
                        f"Run {run_id} failed after {duration:.2f}s: {exc}\n{tb}"
                    )

                row = {
                    "id": run_id,
                    "config_path": str(run_config_path),
                    "output_path": str(result_path),
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

        # Create shared boxplots for this dataset
        shared_dir = ds_folder / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        run_names = list(map(str, range(run_id)))
        if base_cfg["mapping"]["deterministic"]:
            run_names += list(f"{runid}_det" for runid in range(run_id))
        create_shared_boxplots(
            run_names,
            ds_metric_folder,
            shared_dir,
            run_permutation_tests=run_permutation_tests,
        )

    # Create cross-dataset shared boxplots (one box per dataset, best/only run per dataset)
    if len(sc_paths) * len(st_paths) > 1:
        all_metric_folders = []
        all_labels = []
        for sc_path, st_path in itertools.product(sc_paths, st_paths):
            dataset_name = f"{sc_path.parent.name}_{sc_path.stem}__{st_path.parent.name}_{st_path.stem}"
            ds_metric_folder = output_folder / dataset_name / "metrics"
            for run_id_str in map(str, range(total_runs)):
                all_metric_folders.append(ds_metric_folder / run_id_str)
                all_labels.append(f"{dataset_name}/{run_id_str}")
            if base_cfg["mapping"]["deterministic"]:
                for run_id_str in map(str, range(total_runs)):
                    all_metric_folders.append(ds_metric_folder / f"{run_id_str}_det")
                    all_labels.append(f"{dataset_name}/{run_id_str}_det")
        cross_dataset_shared = output_folder / "shared"
        cross_dataset_shared.mkdir(parents=True, exist_ok=True)
        create_shared_boxplots(
            all_labels,
            output_folder,
            cross_dataset_shared,
            run_permutation_tests=run_permutation_tests,
        )


if __name__ == "__main__":

    # 1. Parse Arguments
    parser = argparse.ArgumentParser(
        description="Run AlternativeIdea alignment. Dataset paths and output folders are configured in the experiment YAML."
    )
    parser.add_argument(
        "-c",
        "--experiment_config",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to experiment config YAML. Multiple configs are run sequentially.",
    )
    parser.add_argument(
        "-s",
        "--save_result",
        action="store_true",
        help="Whether to save the predicted GEP to disk",
    )
    parser.add_argument(
        "--run_permutation_tests",
        dest="run_permutation_tests",
        action="store_true",
        help="Whether to run permutation tests or not.",
    )
    parser.add_argument(
        "--logging",
        dest="logging",
        choices=["normal", "verbose"],
        default="normal",
        help="Logging verbosity. Use 'verbose' for more logs.",
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
    for cfg_path in args.experiment_config:
        main(
            cfg_path,
            save_result=args.save_result,
            run_permutation_tests=args.run_permutation_tests,
        )
