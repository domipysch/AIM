"""
Run post-mapping analyses for every pair folder produced by run_all_pairs.py, in parallel.

Usage
-----
    python run_analyses_all_pairs.py \
        --output_dir Data/05_Experiments \
        [--workers 4]

Expected layout
---------------
    <output_dir>/
      pair_0/
        <sc>__<st>/
          summary.csv
          0/  1/  ...
          metrics/
      pair_1/
        ...
"""

import argparse
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from run_analyses_per_k import main as run_analyses_main

logger = logging.getLogger(__name__)


def _analysis_worker(pair_id: int, result_folder: Path) -> list[str]:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger(__name__)
    errors: list[str] = []
    tag = f"[Pair {pair_id:>3}]"
    log.info("%s Starting analysis: %s", tag, result_folder)
    try:
        run_analyses_main(result_folder)
    except Exception as exc:
        msg = f"{tag} run_analyses FAILED: {exc}"
        log.error(msg, exc_info=True)
        errors.append(msg)
    return errors


def main(output_dir: Path, workers: int) -> None:
    output_dir = Path(output_dir)

    pair_dirs = sorted(
        [
            d
            for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("pair_") and d.name[5:].isdigit()
        ],
        key=lambda d: int(d.name[5:]),
    )

    if not pair_dirs:
        logger.error("No pair_* directories found in %s", output_dir)
        sys.exit(1)

    logger.info(
        "Found %d pair folder(s) — running analyses with %d worker(s).",
        len(pair_dirs),
        workers,
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analysis_worker, int(d.name[5:]), d): int(d.name[5:])
            for d in pair_dirs
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
                logger.info("Pair %d analysis done.", pid)

    if errors:
        logger.warning("\n%d error(s) occurred:", len(errors))
        for e in errors:
            logger.warning("  %s", e)
    else:
        logger.info("\nAll analyses completed successfully.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run post-mapping analyses for all pair folders in parallel."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root folder containing pair_0/, pair_1/, ... subdirectories.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    args = parser.parse_args()

    main(output_dir=args.output_dir, workers=args.workers)
