"""Batch runner: reference aligner (Tangram / TACCO / DOT) + metrics for every pair × cell-type granularity.

Run from the repository root with the appropriate conda environment active:
    conda activate tangram_env   # or tacco_env / dot_env
    python -m batch_processing.run_reference_aligner_all_pairs --aligner tangram [options]

For each pair the script iterates over every non-empty CellTypeKey in scRNA/index.csv
(CellTypeKey0, CellTypeKey1, CellTypeKey2) and produces one subtree per granularity:

    <output_dir>/{PairID:03d}_{scName}__{stName}/{cell_type_key}/
        gep_prob.h5ad         <- predicted GEP (probabilistic mapping)
        gep_det.h5ad          <- predicted GEP (deterministic mapping)
        mapping_prob.h5ad
        mapping_det.h5ad
        metrics_prob/         <- metrics evaluated on gep_prob
        metrics_det/          <- metrics evaluated on gep_det
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Callable
import anndata as ad
from src.metrics.run_all_metrics import main as run_all_metrics

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_align_fn(aligner: str) -> Callable:
    """Return an align callable with signature (sc_path, st_path, ct_key, output_folder)."""
    if aligner == "tangram":
        from reference_aligners.run_tangram import tangram_align_data

        def _align(
            sc_path: Path, st_path: Path, ct_key: str, output_folder: Path
        ) -> None:
            tangram_align_data(
                sc_path=sc_path,
                st_path=st_path,
                normalize_and_log=False,
                compute_marker_genes=False,
                map_clusters=True,
                cell_type_key=ct_key,
                output_folder=output_folder,
            )

        return _align

    if aligner == "tacco":
        from reference_aligners.run_tacco import tacco_align_data

        def _align(
            sc_path: Path, st_path: Path, ct_key: str, output_folder: Path
        ) -> None:
            tacco_align_data(
                sc_path=sc_path,
                st_path=st_path,
                map_cell_types=True,
                cell_type_key=ct_key,
                output_folder=output_folder,
            )

        return _align

    if aligner == "dot":
        from reference_aligners.run_dot import dot_align_data

        def _align(
            sc_path: Path, st_path: Path, ct_key: str, output_folder: Path
        ) -> None:
            dot_align_data(
                sc_path=sc_path,
                st_path=st_path,
                cell_type_key=ct_key,
                output_folder=output_folder,
            )

        return _align

    raise ValueError(f"Unknown aligner: {aligner!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reference aligner (Tangram, TACCO, or DOT) and metrics "
        "for every pair in pairs.csv, once per available cell-type granularity."
    )
    parser.add_argument(
        "--aligner",
        choices=["tangram", "tacco", "dot"],
        required=True,
        help="Which aligner to run. Activate the matching conda env before running",
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

    align = _get_align_fn(args.aligner)

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
            gep_prob_path = granularity_dir / "gep_prob.h5ad"
            gep_det_path = granularity_dir / "gep_det.h5ad"
            tag = f"[Pair {pair_id:>3} | {ct_key}]"

            logger.info(f"{tag} Running {args.aligner}: {sc_name}  ×  {st_name}")
            try:
                align(sc_path, st_path, ct_key, granularity_dir)
            except Exception as exc:
                msg = f"{tag} {args.aligner} FAILED: {exc}"
                logger.error(msg)
                errors.append(msg)
                continue

            for gep_path, metrics_subdir in (
                (gep_prob_path, "metrics_prob"),
                (gep_det_path, "metrics_det"),
            ):
                metrics_dir = granularity_dir / metrics_subdir

                if not gep_path.exists():
                    msg = f"{tag} GEP not found, cannot compute {metrics_subdir}: {gep_path}"
                    logger.error(msg)
                    errors.append(msg)
                    continue

                logger.info(f"{tag} Computing {metrics_subdir}")
                try:
                    gep = ad.read_h5ad(gep_path)
                    run_all_metrics(
                        sc_path=sc_path,
                        st_path=st_path,
                        metrics=metrics_dir,
                        result_gep=gep,
                    )
                except Exception as exc:
                    msg = f"{tag} Metrics ({metrics_subdir}) FAILED: {exc}"
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
