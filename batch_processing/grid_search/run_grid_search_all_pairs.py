"""
Generate per-pair experiment configs from the template and run experiments across all pairs.

For each row in pairs.csv the script:
  1. Writes a per-pair config YAML (data/output stripped from the template; hyperparameters kept)
  2. Runs grid_search for all pairs in parallel across GPUs

Post-mapping analyses are handled separately by run_analyses_all_pairs.py.

Usage
-----
    python -m batch_processing.grid_search.run_grid_search_all_pairs \
        -c  batch_processing/grid_search/grid_search_config.yaml \
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
import torch
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_template(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.pop("data", None)
    cfg.pop("output", None)
    return cfg


def _run_pair_worker(
    pair_id: int,
    config_path: Path,
    sc_path: Path,
    st_path: Path,
    pair_output: Path,
    gpu_id: int,
    gpu_limit_gb: int,
) -> list[str]:
    tag = f"[Pair {pair_id:>3} | GPU {gpu_id}]"
    # Spawned processes start with a clean sys.path — re-add the repo root so that
    # bare imports like `from utils import ...` in main.py / grid_search.py resolve.
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Import AFTER setting CUDA_VISIBLE_DEVICES so torch sees the correct device(s)
    from batch_processing.grid_search.grid_search import main as run_experiment_main

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger(__name__)

    errors: list[str] = []

    log.info("%s Starting experiment", tag)
    try:
        run_experiment_main(
            config_path,
            sc_path=sc_path,
            st_path=st_path,
            output_folder=pair_output,
            gpu_limit_gb=gpu_limit_gb,
        )
    except Exception as exc:
        msg = f"{tag} run_experiment FAILED: {exc}"
        log.error(msg)
        errors.append(msg)
        return errors

    return errors


def main(
    experiment_config: Path,
    pairs_csv: Path,
    sc_dir: Path,
    st_dir: Path,
    output_dir: Path,
    gpus: list[int],
    gpu_limit_gb: int = 6,
    first_st_per_sc: bool = False,
) -> None:
    experiment_config = Path(experiment_config)
    pairs_csv = Path(pairs_csv)
    sc_dir = Path(sc_dir)
    st_dir = Path(st_dir)
    output_dir = Path(output_dir)

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
    n_workers = len(gpus)

    pairs = pd.read_csv(pairs_csv)
    if first_st_per_sc:
        # Keep only the first ST slice per scRNA dataset (min PairID per scName).
        pairs = pairs.loc[pairs.groupby("scName")["PairID"].idxmin()].reset_index(
            drop=True
        )
        logger.info(
            "--first_st_per_sc: restricted to %d pair(s) (one per scRNA): %s",
            len(pairs),
            pairs["PairID"].tolist(),
        )
    template = _load_template(experiment_config)

    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    pair_ids: list[int] = []
    config_paths: list[Path] = []
    sc_paths: list[Path] = []
    st_paths: list[Path] = []
    pair_outputs: list[Path] = []

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

        config_path = config_dir / f"pair_{pair_id}.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(template, f, sort_keys=False, allow_unicode=True)
        logger.info("Config written → %s", config_path)

        pair_ids.append(pair_id)
        config_paths.append(config_path)
        sc_paths.append(sc_path.resolve())
        st_paths.append(st_path.resolve())
        pair_outputs.append(pair_output.resolve())

    if not config_paths:
        logger.error("No valid pairs found — aborting.")
        sys.exit(1)

    logger.info(
        "=== Running %d pair(s) across %d GPU(s): %s ===",
        len(config_paths),
        len(gpus),
        gpus,
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_pair_worker,
                pair_id,
                cfg_path,
                sc_path,
                st_path,
                pair_output,
                gpus[i % len(gpus)],
                gpu_limit_gb,
            ): pair_id
            for i, (pair_id, cfg_path, sc_path, st_path, pair_output) in enumerate(
                zip(pair_ids, config_paths, sc_paths, st_paths, pair_outputs)
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
        "-c",
        "--experiment_config",
        type=Path,
        required=True,
        help="Path to experiment config YAML (template applied to every pair).",
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
        "--gpu_limit_gb",
        dest="gpu_limit_gb",
        type=int,
        default=48,
        help="GPU memory limit in GB. Abort if estimated usage exceeds this value (default: 6).",
    )
    parser.add_argument(
        "--first_st_per_sc",
        action="store_true",
        help="Run only the first ST slice per scRNA dataset (min PairID per scName). "
        "Reduces the 213 pairs to one per reference (10).",
    )
    args = parser.parse_args()

    main(
        experiment_config=args.experiment_config,
        pairs_csv=args.pairs_csv,
        sc_dir=args.sc_dir,
        st_dir=args.st_dir,
        output_dir=args.output_dir,
        gpus=args.gpus,
        gpu_limit_gb=args.gpu_limit_gb,
        first_st_per_sc=args.first_st_per_sc,
    )
