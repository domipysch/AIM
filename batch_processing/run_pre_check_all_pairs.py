"""Batch runner: runs pre_check for every pair in pairs.csv.

Run from the repository root with the aim_env active:
    conda activate aim_env
    python run_pre_check_all_pairs.py [options]

scRNA AnnData objects are cached in memory — each dataset is loaded only once
even when it appears in multiple pairs (e.g. 05_mouse-embryo × 6 ST slices).
"""

import argparse
import concurrent.futures
import csv
import logging
from pathlib import Path
import scanpy as sc
from pre_check.pre_check import run_pre_check

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_pair(
    pair: dict,
    args: argparse.Namespace,
    sc_cache: dict,
) -> str | None:
    """Run pre_check for one pair. Returns None on success, error string on failure."""
    pair_id = int(pair["PairID"])
    sc_name = pair["scName"]
    st_name = pair["stName"]

    out_dir = args.output_dir / f"{pair_id:03d}_{sc_name}__{st_name}"
    sc_path = args.sc_dir / f"{sc_name}.h5ad"
    st_path = args.st_dir / f"{st_name}.h5ad"

    missing = [p for p in (sc_path, st_path) if not p.exists()]
    if missing:
        msg = f"[Pair {pair_id:>3}] Missing files: {[str(p) for p in missing]}"
        logger.error(msg)
        return msg

    logger.info(f"[Pair {pair_id:>3}] {sc_name}  ×  {st_name}")

    sc_adata = sc_cache[sc_name]

    logger.info(f"  Loading ST   : {st_path.name}")
    st_adata = sc.read_h5ad(st_path)

    try:
        run_pre_check(
            sc_adata=sc_adata,
            st_adata=st_adata,
            output_dir=out_dir,
            leiden_resolution=args.leiden_resolution,
        )
        logger.info(f"[Pair {pair_id:>3}] Done → {out_dir}")
        return None
    except Exception as exc:
        msg = f"[Pair {pair_id:>3}] FAILED: {exc}"
        logger.error(msg)
        return msg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pre_check for every pair listed in pairs.csv."
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
        help="Folder containing scRNA .h5ad files",
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
        "--leiden_resolution",
        type=float,
        default=0.5,
        help="Leiden clustering resolution  (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers (default: cpu count)",
    )
    args = parser.parse_args()

    # Load pairs
    with open(args.pairs_csv, newline="") as fh:
        all_pairs = list(csv.DictReader(fh))

    logger.info(f"Loaded {len(all_pairs)} pairs from {args.pairs_csv}")

    # Pre-load scRNA cache sequentially so the cache is read-only during parallel execution
    sc_cache: dict[str, sc.AnnData] = {}
    for pair in all_pairs:
        sc_name = pair["scName"]
        if sc_name not in sc_cache:
            sc_path = args.sc_dir / f"{sc_name}.h5ad"
            logger.info(f"  Loading scRNA: {sc_path.name}")
            sc_cache[sc_name] = sc.read_h5ad(sc_path)

    errors: list[str] = []

    if args.workers > 1:
        logger.info(
            f"Running {len(all_pairs)} pairs in parallel (workers={args.workers})"
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futs = {
                executor.submit(process_pair, pair, args, sc_cache): pair
                for pair in all_pairs
            }
            for fut in concurrent.futures.as_completed(futs):
                result = fut.result()
                if result:
                    errors.append(result)
    else:
        for pair in all_pairs:
            result = process_pair(pair, args, sc_cache)
            if result:
                errors.append(result)

    # Summary
    if errors:
        logger.warning(f"\n{len(errors)} pair(s) failed:")
        for e in errors:
            logger.warning(f"  {e}")
    else:
        logger.info("\nAll pairs completed successfully.")


if __name__ == "__main__":
    main()
