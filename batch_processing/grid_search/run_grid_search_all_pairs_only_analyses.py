"""
Re-run ONLY the post-mapping analyses for all already-computed dataset pairs.

This is the analysis-only companion to ``run_grid_search_all_pairs.py``. It does
**not** recompute the mapping/training: for every pair it reuses each run's stored
B/C assignment matrices (``<pair>/<run>/intermediate/``) and re-runs

  * the per-run analysis + per-K report (``<pair>/<run>/analysis/``), and
  * the cross-K summary analysis (``<pair>/analysis_overview.csv`` + summary PDF).

Mapping outputs, per-run metrics and shared boxplots are left untouched. Use it
after changing anything in ``evaluate_k`` (plots, reports, analysis metrics) to
refresh the figures/tables without paying for the GPU mapping again.

Because no mapping is recomputed, this runs on CPU only (CUDA is disabled in the
workers); parallelism is over CPU worker processes, one pair per worker.

Usage
-----
    python -m batch_processing.grid_search.run_grid_search_all_pairs_only_analyses \
        -c  batch_processing/grid_search/grid_search_config.yaml \
        --pairs_csv  Data/01_Datasets/pairs.csv \
        --sc_dir     Data/01_Datasets/scRNA \
        --st_dir     Data/01_Datasets/ST \
        --output_dir Data/05_Experiment \
        --workers 4

The ``--output_dir`` must be the same folder the mapping wrote to. Pairs whose
output folder does not exist yet are skipped (run the mapping for them first).
The Leiden resolution is taken from each pair's saved ``experiment_config.yml``
when present, otherwise from the ``-c`` template.
"""

import argparse
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _run_pair_analyses_worker(
    pair_id: int,
    experiment_config: Path,
    sc_path: Path,
    st_path: Path,
    pair_output: Path,
) -> list[str]:
    tag = f"[Pair {pair_id:>3}]"
    # Analyses are CPU-only — hide CUDA so importing torch never grabs a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from batch_processing.grid_search.grid_search import run_analyses_only

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger(__name__)

    # Prefer the config the mapping actually used (copied into the pair folder).
    saved_cfg = pair_output / "experiment_config.yml"
    config_path = saved_cfg if saved_cfg.exists() else experiment_config

    log.info("%s Re-running analyses (config: %s)", tag, config_path)
    try:
        run_analyses_only(
            experiment_config=config_path,
            sc_path=sc_path,
            st_path=st_path,
            output_folder=pair_output,
        )
    except Exception as exc:
        msg = f"{tag} analyses FAILED: {exc}"
        log.error(msg)
        return [msg]

    return []


def main(
    experiment_config: Path,
    pairs_csv: Path,
    sc_dir: Path,
    st_dir: Path,
    output_dir: Path,
    workers: int = 4,
) -> None:
    experiment_config = Path(experiment_config)
    pairs_csv = Path(pairs_csv)
    sc_dir = Path(sc_dir)
    st_dir = Path(st_dir)
    output_dir = Path(output_dir)

    pairs = pd.read_csv(pairs_csv)

    pair_ids: list[int] = []
    sc_paths: list[Path] = []
    st_paths: list[Path] = []
    pair_outputs: list[Path] = []

    for _, row in pairs.iterrows():
        pair_id = int(row["PairID"])
        sc_path = sc_dir / f"{row['scName']}.h5ad"
        st_path = st_dir / f"{row['stName']}.h5ad"
        pair_output = output_dir / f"pair_{pair_id}"

        if not pair_output.is_dir():
            logger.warning(
                "Pair %d: output folder missing — skipping (run the mapping first): %s",
                pair_id,
                pair_output,
            )
            continue
        if not sc_path.exists():
            logger.warning("SC file not found — skipping pair %d: %s", pair_id, sc_path)
            continue
        if not st_path.exists():
            logger.warning("ST file not found — skipping pair %d: %s", pair_id, st_path)
            continue

        pair_ids.append(pair_id)
        sc_paths.append(sc_path.resolve())
        st_paths.append(st_path.resolve())
        pair_outputs.append(pair_output.resolve())

    if not pair_ids:
        logger.error("No already-computed pairs found under %s — aborting.", output_dir)
        sys.exit(1)

    n_workers = max(1, min(workers, len(pair_ids)))
    logger.info(
        "=== Re-running analyses for %d pair(s) across %d worker(s) ===",
        len(pair_ids),
        n_workers,
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_pair_analyses_worker,
                pair_id,
                experiment_config,
                sc_path,
                st_path,
                pair_output,
            ): pair_id
            for pair_id, sc_path, st_path, pair_output in zip(
                pair_ids, sc_paths, st_paths, pair_outputs
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
                logger.info("Pair %d analyses done.", pid)

    if errors:
        logger.warning("\n%d error(s) occurred:", len(errors))
        for e in errors:
            logger.warning("  %s", e)
    else:
        logger.info("\nAll pair analyses completed successfully.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Re-run only the post-mapping analyses (per-run + summary) for all "
        "already-computed pairs, without recomputing the mapping."
    )
    parser.add_argument(
        "-c",
        "--experiment_config",
        type=Path,
        required=True,
        help="Experiment config YAML template (fallback for the Leiden resolution; "
        "each pair's saved experiment_config.yml is preferred when present).",
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
        help="Root folder containing the per-pair outputs from the mapping run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of pairs to analyse in parallel (CPU processes). Default: 4. "
        "Each worker is itself multi-threaded (scanpy/BLAS), so keep this modest.",
    )
    args = parser.parse_args()

    main(
        experiment_config=args.experiment_config,
        pairs_csv=args.pairs_csv,
        sc_dir=args.sc_dir,
        st_dir=args.st_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )
