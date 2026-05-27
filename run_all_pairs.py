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
        --output_dir Data/05_Experiments

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
import sys
from pathlib import Path

import pandas as pd
import yaml

from run_experiment import main as run_experiment_main
from run_analyses_per_k import main as run_analyses_main

logger = logging.getLogger(__name__)

_TEMPLATE_CONFIG = Path(__file__).parent / "experiment_config.yaml"


def _load_template() -> dict:
    with open(_TEMPLATE_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.pop("data", None)
    cfg.pop("output", None)
    return cfg


def main(
    pairs_csv: Path,
    sc_dir: Path,
    st_dir: Path,
    output_dir: Path,
    save_result: bool = False,
    run_permutation_tests: bool = False,
    no_gpu_limit: bool = False,
) -> None:
    pairs_csv = Path(pairs_csv)
    sc_dir = Path(sc_dir)
    st_dir = Path(st_dir)
    output_dir = Path(output_dir)

    pairs = pd.read_csv(pairs_csv)
    template = _load_template()

    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

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

        config_paths.append(config_path)
        result_folders.append(pair_output)

    if not config_paths:
        logger.error("No valid pairs found — aborting.")
        sys.exit(1)

    # --- run experiments ---
    logger.info("=== Running experiments for %d pair(s) ===", len(config_paths))
    for cfg_path in config_paths:
        run_experiment_main(
            cfg_path,
            save_result=save_result,
            run_permutation_tests=run_permutation_tests,
            no_gpu_limit=no_gpu_limit,
        )

    # --- run analyses ---
    logger.info("=== Running analyses for %d pair(s) ===", len(result_folders))
    for result_folder in result_folders:
        run_analyses_main(result_folder)


if __name__ == "__main__":
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
        "--save_result",
        action="store_true",
        help="Save the predicted GEP to disk (passed through to run_experiment).",
    )
    parser.add_argument(
        "--run_permutation_tests",
        action="store_true",
        help="Run permutation tests (passed through to run_experiment).",
    )
    parser.add_argument(
        "--no_gpu_limit",
        dest="no_gpu_limit",
        action="store_true",
        default=False,
        help=f"Bypass the GPU memory guard and run regardless of estimated memory usage.",
    )
    args = parser.parse_args()

    main(
        pairs_csv=args.pairs_csv,
        sc_dir=args.sc_dir,
        st_dir=args.st_dir,
        output_dir=args.output_dir,
        save_result=args.save_result,
        run_permutation_tests=args.run_permutation_tests,
    )
