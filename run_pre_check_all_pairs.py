"""Batch runner: runs pre_check for every pair in pairs.csv.

Run from the repository root with the alternative_idea_env active:
    conda activate alternative_idea_env
    python run_pre_check_all_pairs.py [options]

scRNA AnnData objects are cached in memory — each dataset is loaded only once
even when it appears in multiple pairs (e.g. 05_mouse-embryo × 6 ST slices).
"""

import argparse
import csv
import logging
from pathlib import Path

import scanpy as sc

from src.pre_check import run_pre_check

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
    "--permutation_test",
    action="store_true",
    help="Run the permutation test (slow)",
)
parser.add_argument(
    "--n_permutations",
    type=int,
    default=200,
    help="Number of permutations  (default: %(default)s)",
)
parser.add_argument(
    "--no_skip",
    action="store_true",
    help="Re-run pairs that already have a results.json (default: skip them)",
)
parser.add_argument(
    "--pair_ids",
    type=int,
    nargs="+",
    metavar="ID",
    help="Only run these PairIDs (space-separated); omit to run all",
)
args = parser.parse_args()

skip_existing = not args.no_skip

# ---------------------------------------------------------------------------
# Load pairs
# ---------------------------------------------------------------------------
with open(args.pairs_csv, newline="") as fh:
    pairs = list(csv.DictReader(fh))

logger.info(f"Loaded {len(pairs)} pairs from {args.pairs_csv}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
sc_cache: dict[str, sc.AnnData] = {}  # sc_name → AnnData (avoids redundant loads)
errors: list[str] = []

for pair in pairs:
    pair_id = int(pair["PairID"])
    sc_name = pair["scName"]
    st_name = pair["stName"]

    if args.pair_ids and pair_id not in args.pair_ids:
        continue

    out_dir = args.output_dir / f"{pair_id:03d}_{sc_name}__{st_name}"

    if skip_existing and (out_dir / "results.json").exists():
        logger.info(f"[Pair {pair_id:>3}] Skipping — results.json already exists")
        continue

    sc_path = args.sc_dir / f"{sc_name}.h5ad"
    st_path = args.st_dir / f"{st_name}.h5ad"

    missing = [p for p in (sc_path, st_path) if not p.exists()]
    if missing:
        msg = f"[Pair {pair_id:>3}] Missing files: {[str(p) for p in missing]}"
        logger.error(msg)
        errors.append(msg)
        continue

    logger.info(f"[Pair {pair_id:>3}] {sc_name}  ×  {st_name}")

    # Load scRNA (cached)
    if sc_name not in sc_cache:
        logger.info(f"  Loading scRNA: {sc_path.name}")
        sc_cache[sc_name] = sc.read_h5ad(sc_path)
    sc_adata = sc_cache[sc_name]

    # Load ST (not cached — each slice is unique)
    logger.info(f"  Loading ST   : {st_path.name}")
    st_adata = sc.read_h5ad(st_path)

    try:
        run_pre_check(
            sc_adata=sc_adata,
            st_adata=st_adata,
            output_dir=out_dir,
            leiden_resolution=args.leiden_resolution,
            run_permutation_test=args.permutation_test,
            n_permutations=args.n_permutations,
        )
        logger.info(f"[Pair {pair_id:>3}] Done → {out_dir}")
    except Exception as exc:
        msg = f"[Pair {pair_id:>3}] FAILED: {exc}"
        logger.error(msg)
        errors.append(msg)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if errors:
    logger.warning(f"\n{len(errors)} pair(s) failed:")
    for e in errors:
        logger.warning(f"  {e}")
else:
    logger.info("\nAll pairs completed successfully.")
