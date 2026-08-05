"""Batch runner: a reference aligner for every pair × cell-type granularity.

Run from the repository root in ``aim_env`` (the aligner itself runs in its own
env via ``conda run``, so only ``conda`` on PATH is required):
    conda activate aim_env
    python -m reference_aligners.run_reference_aligner_all_pairs --aligner tangram [options]

For each pair the script iterates over every non-empty CellTypeKey in scRNA/index.csv
(CellTypeKey0, CellTypeKey1, CellTypeKey2) and produces one subtree per granularity:

    <output_dir>/{PairID:03d}_{scName}__{stName}/{cell_type_key}/
        mapping_prob.h5ad     <- spots x type mapping, var_names = cell type names
        analysis/data/        <- mapping analysis metrics (always run; no figures)

The set of available aligners comes from ``reference_aligners.registry``; adding
one there makes it selectable here automatically.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

# The mapping analysis step imports `metrics.*` / `utils`, which live under
# src/ — add it to sys.path here so this script works regardless of whether
# the caller remembers to set PYTHONPATH=src.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from reference_aligners.mapping_analysis.analyze import analyze_mapping
from reference_aligners.registry import REFERENCE_ALIGNERS, run_aligner

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reference aligner for every pair in pairs.csv, "
        "once per available cell-type granularity."
    )
    parser.add_argument(
        "--aligner",
        choices=list(REFERENCE_ALIGNERS),
        required=True,
        help="Which reference aligner to run (each runs in its own conda env "
        "via `conda run`; run this script from aim_env).",
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
    args = parser.parse_args()

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

    errors: list[str] = []

    for pair in pairs:
        pair_id = int(pair["PairID"])

        sc_name = pair["scName"]
        st_name = pair["stName"]
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
            granularity_dir = pair_dir / ct_key
            mapping_prob_path = granularity_dir / "mapping_prob.h5ad"
            tag = f"[Pair {pair_id:>3} | {ct_key}]"

            logger.info(f"{tag} Running {args.aligner}: {sc_name}  ×  {st_name}")
            try:
                run_aligner(args.aligner, sc_path, st_path, granularity_dir, ct_key)
            except Exception as exc:
                msg = f"{tag} {args.aligner} FAILED: {exc}"
                logger.error(msg)
                errors.append(msg)
                continue

            if not mapping_prob_path.exists():
                msg = (
                    f"{tag} mapping_prob.h5ad not found after run: {mapping_prob_path}"
                )
                logger.error(msg)
                errors.append(msg)
                continue

            logger.info(f"{tag} Running mapping analysis")
            try:
                analyze_mapping(sc_path, st_path, granularity_dir, cell_type_key=ct_key)
            except Exception as exc:
                msg = f"{tag} mapping analysis FAILED: {exc}"
                logger.error(msg)
                errors.append(msg)

        logger.info(f"[Pair {pair_id:>3}] Done → {pair_dir}")

    if errors:
        logger.warning(f"\n{len(errors)} error(s) occurred:")
        for e in errors:
            logger.warning(f"  {e}")
    else:
        logger.info("\nAll pairs completed successfully.")


if __name__ == "__main__":
    main()
