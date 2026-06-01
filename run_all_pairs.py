"""
Generate per-pair experiment configs from the template and run the full pipeline.

For each row in pairs.csv the script:
  1. Writes a config YAML (data + output filled in, rest from experiment_config.yaml)
  2. Runs run_experiment for all pairs
  3. Runs run_analyses_per_k for all pairs

Usage
-----
    python run_all_pairs.py \
        --pairs_csv  Data/01_Datasets/pairs.csv \
        --sc_dir     Data/01_Datasets/scRNA \
        --st_dir     Data/01_Datasets/ST \
        --output_dir Data/05_Experiments \
        --gpus 0 1 2 3

Output layout
-------------
    <output_dir>/
      configs/
        pair_0.yaml
        pair_1.yaml
        ...
      pair_0/
        <sc_stem>__<st_stem>/   ← produced by run_experiment
          ...
      pair_1/
        ...
"""

import argparse
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from run_experiment import main as run_experiment_main
from run_analyses_per_k import main as run_analyses_main

logger = logging.getLogger(__name__)

_TEMPLATE_CONFIG = Path(__file__).parent / "experiment_config.yaml"


def _count_runs(node) -> int:
    """Return the total number of grid-search combinations for a config node."""
    if isinstance(node, list):
        return len(node) if node else 1
    if isinstance(node, dict):
        total = 1
        for v in node.values():
            total *= _count_runs(v)
        return total
    return 1  # scalar


def _pair_is_complete(
    pair_output: Path, sc_path: Path, st_path: Path, expected_runs: int
) -> bool:
    """Return True if summary.csv exists with exactly expected_runs ok rows."""
    dataset_name = (
        f"{sc_path.parent.name}_{sc_path.stem}"
        f"__{st_path.parent.name}_{st_path.stem}"
    )
    summary_path = pair_output / dataset_name / "summary.csv"
    if not summary_path.exists():
        return False
    try:
        df = pd.read_csv(summary_path)
        print(expected_runs)
        return len(df) == expected_runs and (df["status"] == "ok").all()
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, KeyError):
        return False


def _load_template() -> dict:
    with open(_TEMPLATE_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.pop("data", None)
    cfg.pop("output", None)
    return cfg


def _run_pair_worker(
    pair_id: int,
    config_path: Path,
    result_folder: Path,
    gpu_id: int,
    save_result: bool,
    run_permutation_tests: bool,
    no_gpu_limit: bool,
) -> list[str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger(__name__)

    errors: list[str] = []
    tag = f"[Pair {pair_id:>3} | GPU {gpu_id}]"

    log.info("%s Starting experiment", tag)
    try:
        run_experiment_main(
            config_path,
            save_result=save_result,
            run_permutation_tests=run_permutation_tests,
            no_gpu_limit=no_gpu_limit,
        )
    except Exception as exc:
        msg = f"{tag} run_experiment FAILED: {exc}"
        log.error(msg)
        errors.append(msg)
        return errors

    log.info("%s Starting analysis", tag)
    try:
        run_analyses_main(result_folder)
    except Exception as exc:
        msg = f"{tag} run_analyses FAILED: {exc}"
        log.error(msg)
        errors.append(msg)

    return errors


def main(
    pairs_csv: Path,
    sc_dir: Path,
    st_dir: Path,
    output_dir: Path,
    gpus: list[int],
    save_result: bool = False,
    run_permutation_tests: bool = False,
    no_gpu_limit: bool = False,
) -> None:
    pairs_csv = Path(pairs_csv)
    sc_dir = Path(sc_dir)
    st_dir = Path(st_dir)
    output_dir = Path(output_dir)

    import torch

    n_cuda = torch.cuda.device_count()
    invalid = [g for g in gpus if g >= n_cuda]
    if invalid:
        logger.error(
            "Invalid GPU ID(s) %s — only %d CUDA device(s) available (IDs 0–%d).",
            invalid,
            n_cuda,
            n_cuda - 1,
        )
        sys.exit(1)

    pairs = pd.read_csv(pairs_csv)
    template = _load_template()
    expected_runs = _count_runs(template)

    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    pair_ids: list[int] = []
    config_paths: list[Path] = []
    result_folders: list[Path] = []

    # --- generate one config per pair ---
    for _, row in pairs.iterrows():
        pair_id = int(row["PairID"])
        sc_path = sc_dir / f"{row['scName']}.h5ad"
        st_path = st_dir / f"{row['stName']}.h5ad"

        if not sc_path.exists():
            logger.warning("SC file not found — skipping pair %d: %s", pair_id, sc_path)
            continue
        if not st_path.exists():
            logger.warning("ST file not found — skipping pair %d: %s", pair_id, st_path)
            continue

        pair_output = output_dir / f"pair_{pair_id}"

        if _pair_is_complete(pair_output, sc_path, st_path, expected_runs):
            logger.info("Pair %d already complete — skipping.", pair_id)
            continue

        cfg = {
            "data": {
                "sc_paths": [str(sc_path.resolve())],
                "st_paths": [str(st_path.resolve())],
            },
            "output": {
                "output_folder": str(pair_output.resolve()),
            },
            **template,
        }

        config_path = config_dir / f"pair_{pair_id}.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        logger.info("Config written → %s", config_path)

        pair_ids.append(pair_id)
        config_paths.append(config_path)
        result_folders.append(pair_output)

    if not config_paths:
        logger.error("No valid pairs found — aborting.")
        sys.exit(1)

    logger.info(
        "=== Running %d pair(s) across %d GPU(s): %s ===",
        len(config_paths),
        len(gpus),
        gpus,
    )

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                _run_pair_worker,
                pair_id,
                cfg_path,
                result_folder,
                gpus[i % len(gpus)],
                save_result,
                run_permutation_tests,
                no_gpu_limit,
            ): pair_id
            for i, (pair_id, cfg_path, result_folder) in enumerate(
                zip(pair_ids, config_paths, result_folders)
            )
        }
        errors: list[str] = []
        for future in as_completed(futures):
            pid = futures[future]
            try:
                errors.extend(future.result())
            except Exception as exc:
                msg = f"[Pair {pid:>3}] Unexpected worker error: {exc}"
                logger.error(msg)
                errors.append(msg)
            else:
                logger.info("Pair %d done.", pid)

    if errors:
        logger.warning("\n%d error(s) occurred:", len(errors))
        for e in errors:
            logger.warning("  %s", e)
    else:
        logger.info("\nAll pairs completed successfully.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate per-pair configs and run the full experiment + analysis pipeline."
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        required=True,
        help="Path to pairs.csv",
    )
    parser.add_argument(
        "--sc_dir",
        type=Path,
        required=True,
        help="Folder containing scRNA .h5ad files and index.csv",
    )
    parser.add_argument(
        "--st_dir",
        type=Path,
        required=True,
        help="Folder containing ST .h5ad files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root folder for all pair outputs",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        metavar="ID",
        help="GPU IDs to use (space-separated). One pair runs per GPU in parallel. "
        "Default: 0 (single GPU, sequential).",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="Save the predicted GEP to disk (passed through to run_experiment).",
    )
    parser.add_argument(
        "--run_permutation_tests",
        action="store_true",
        default=False,
        help="Run permutation tests (passed through to run_experiment).",
    )
    parser.add_argument(
        "--no_gpu_limit",
        dest="no_gpu_limit",
        action="store_true",
        default=False,
        help="Bypass the GPU memory guard and run regardless of estimated memory usage.",
    )
    args = parser.parse_args()

    main(
        pairs_csv=args.pairs_csv,
        sc_dir=args.sc_dir,
        st_dir=args.st_dir,
        output_dir=args.output_dir,
        gpus=args.gpus,
        save_result=args.save_result,
        run_permutation_tests=args.run_permutation_tests,
        no_gpu_limit=args.no_gpu_limit,
    )
