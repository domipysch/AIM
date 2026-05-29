"""Batch runner: DOT alignment + metrics for every pair × cell-type granularity.

Run from the repository root with the dot_env active:
    conda activate dot_env
    python -m src.aligners.run_dot_all_pairs [options]

For each pair the script iterates over every non-empty CellTypeKey in scRNA/index.csv
(CellTypeKey0, CellTypeKey1, CellTypeKey2) and produces one subtree per granularity:

    <output_dir>/{PairID:03d}_{scName}__{stName}/{cell_type_key}/
        gep_prob.h5ad         ← predicted GEP (probabilistic mapping)
        gep_det.h5ad          ← predicted GEP (deterministic mapping)
        mapping_prob.h5ad
        mapping_det.h5ad
        metrics_prob/         ← metrics evaluated on gep_prob
        metrics_det/          ← metrics evaluated on gep_det
"""

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import anndata as ad

from .run_dot import dot_align_data
from ..metrics.run_all_metrics import main as run_all_metrics

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run_granularity(
    pair_id: int,
    sc_name: str,
    st_name: str,
    sc_path: Path,
    st_path: Path,
    ct_key: str,
    granularity_dir: Path,
    skip_existing: bool,
    run_permutation_tests: bool,
) -> list[str]:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger(__name__)

    errors: list[str] = []
    tag = f"[Pair {pair_id:>3} | {ct_key}]"

    gep_prob_path = granularity_dir / "gep_prob.h5ad"
    gep_det_path = granularity_dir / "gep_det.h5ad"

    # -------------------------------------------------------------------
    # DOT alignment
    # -------------------------------------------------------------------
    if skip_existing and gep_prob_path.exists() and gep_det_path.exists():
        log.info(f"{tag} Skipping DOT — GEP outputs already exist")
    else:
        log.info(f"{tag} Running DOT: {sc_name}  ×  {st_name}")
        try:
            dot_align_data(
                sc_path=sc_path,
                st_path=st_path,
                cell_type_key=ct_key,
                output_folder=granularity_dir,
            )
        except Exception as exc:
            msg = f"{tag} DOT FAILED: {exc}"
            log.error(msg)
            errors.append(msg)
            return errors

    # -------------------------------------------------------------------
    # Metrics — run for both probabilistic and deterministic GEPs
    # -------------------------------------------------------------------
    for gep_path, metrics_subdir in (
        (gep_prob_path, "metrics_prob"),
        (gep_det_path, "metrics_det"),
    ):
        metrics_dir = granularity_dir / metrics_subdir

        if skip_existing and metrics_dir.exists() and any(metrics_dir.iterdir()):
            log.info(f"{tag} Skipping {metrics_subdir} — already populated")
            continue

        if not gep_path.exists():
            msg = f"{tag} GEP not found, cannot compute {metrics_subdir}: {gep_path}"
            log.error(msg)
            errors.append(msg)
            continue

        log.info(f"{tag} Computing {metrics_subdir}")
        try:
            gep = ad.read_h5ad(gep_path)
            run_all_metrics(
                sc_path=sc_path,
                st_path=st_path,
                metrics=metrics_dir,
                result_gep=gep,
                run_permutation_tests=run_permutation_tests,
            )
        except Exception as exc:
            msg = f"{tag} Metrics ({metrics_subdir}) FAILED: {exc}"
            log.error(msg)
            errors.append(msg)

    return errors


def main() -> None:
    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Run DOT alignment and metrics for every pair in pairs.csv, "
        "once per available cell-type granularity."
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
        "--permutation_tests",
        action="store_true",
        default=False,
        help="Run permutation tests during metrics computation (slow)",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Re-run runs that already have both GEP outputs (default: skip them)",
    )
    parser.add_argument(
        "--pair_ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Only run these PairIDs (space-separated); omit to run all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 4)",
    )
    args = parser.parse_args()

    skip_existing = not args.no_skip

    # ---------------------------------------------------------------------------
    # Load index tables
    # ---------------------------------------------------------------------------
    with open(args.pairs_csv, newline="") as fh:
        pairs = list(csv.DictReader(fh))

    with open(args.sc_dir / "index.csv", newline="") as fh:
        sc_index = {row["Name"]: row for row in csv.DictReader(fh)}

    logger.info(f"Loaded {len(pairs)} pairs from {args.pairs_csv}")

    def cell_type_keys(sc_name: str) -> list[str]:
        row = sc_index.get(sc_name, {})
        return [
            row[col]
            for col in ("CellTypeKey0", "CellTypeKey1", "CellTypeKey2")
            if row.get(col)
        ]

    # ---------------------------------------------------------------------------
    # Build task list
    # ---------------------------------------------------------------------------
    tasks: list[tuple] = []
    errors: list[str] = []

    for pair in pairs:
        pair_id = int(pair["PairID"])
        sc_name = pair["scName"]
        st_name = pair["stName"]

        if args.pair_ids and pair_id not in args.pair_ids:
            continue

        sc_path = args.sc_dir / f"{sc_name}.h5ad"
        st_path = args.st_dir / f"{st_name}.h5ad"

        missing = [p for p in (sc_path, st_path) if not p.exists()]
        if missing:
            msg = f"[Pair {pair_id:>3}] Missing files: {[str(p) for p in missing]}"
            logger.error(msg)
            errors.append(msg)
            continue

        keys = cell_type_keys(sc_name)
        if not keys:
            msg = f"[Pair {pair_id:>3}] No CellTypeKey found in index for '{sc_name}' — skipping"
            logger.warning(msg)
            errors.append(msg)
            continue

        pair_dir = args.output_dir / f"{pair_id:03d}_{sc_name}__{st_name}"
        for ct_key in keys:
            tasks.append(
                (
                    pair_id,
                    sc_name,
                    st_name,
                    sc_path,
                    st_path,
                    ct_key,
                    pair_dir / ct_key,
                    skip_existing,
                    args.permutation_tests,
                )
            )

    logger.info(f"Submitting {len(tasks)} task(s) with {args.workers} worker(s)")

    # ---------------------------------------------------------------------------
    # Run in parallel
    # ---------------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_granularity, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            pair_id, _, _, _, _, ct_key, granularity_dir = task[:7]
            try:
                errors.extend(future.result())
            except Exception as exc:
                msg = f"[Pair {pair_id:>3} | {ct_key}] Unexpected worker error: {exc}"
                logger.error(msg)
                errors.append(msg)
            else:
                logger.info(f"[Pair {pair_id:>3} | {ct_key}] Done → {granularity_dir}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    if errors:
        logger.warning(f"\n{len(errors)} error(s) occurred:")
        for e in errors:
            logger.warning(f"  {e}")
    else:
        logger.info("\nAll pairs completed successfully.")


if __name__ == "__main__":
    main()
