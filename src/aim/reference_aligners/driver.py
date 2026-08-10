"""Single entry point for running a reference aligner - one pair or a whole
``pairs.csv`` - with its canonical baseline settings.

Both drivers dispatch through :func:`aim.reference_aligners.registry.run_aligner`,
which runs the chosen aligner in its own conda env via ``conda run``; this module
adds the per-pair / per-granularity orchestration and the post-mapping analysis.
``aim map-annotation`` (see :mod:`aim.cli`) calls these functions.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from aim.reference_aligners.mapping_analysis.analyze import analyze_mapping
from aim.reference_aligners.registry import run_aligner

logger = logging.getLogger(__name__)


def align_single(
    aligner: str,
    sc_path: Path,
    st_path: Path,
    output_dir: Path,
    cell_type_key: str,
) -> None:
    """Run one reference alignment for a single sc/st pair, then analyse it.

    Writes ``mapping_prob.h5ad`` and ``analysis/data/`` under ``output_dir``.
    """
    output_dir = Path(output_dir)
    logger.info(
        "Running %s: %s x %s (cell_type_key=%s)",
        aligner,
        sc_path,
        st_path,
        cell_type_key,
    )
    run_aligner(aligner, sc_path, st_path, output_dir, cell_type_key)
    logger.info("Running mapping analysis")
    analyze_mapping(sc_path, st_path, output_dir, cell_type_key=cell_type_key)


def align_batch(
    aligner: str,
    pairs_csv: Path,
    sc_dir: Path,
    st_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Run a reference aligner for every pair in ``pairs.csv``, once per available
    cell-type granularity. Returns the list of error messages (empty on success).

    For each pair the scRNA ``index.csv`` row provides the cell-type keys
    (``CellTypeKey0/1/2``); each produces one subtree::

        <output_dir>/{PairID:03d}_{scName}__{stName}/{cell_type_key}/
            mapping_prob.h5ad
            analysis/data/
    """
    pairs_csv, sc_dir, st_dir, output_dir = map(
        Path, (pairs_csv, sc_dir, st_dir, output_dir)
    )

    with open(pairs_csv, newline="") as fh:
        pairs = list(csv.DictReader(fh))
    with open(sc_dir / "index.csv", newline="") as fh:
        sc_index = {row["Name"]: row for row in csv.DictReader(fh)}

    logger.info("Loaded %d pairs from %s", len(pairs), pairs_csv)

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
        sc_path = sc_dir / f"{sc_name}.h5ad"
        st_path = st_dir / f"{st_name}.h5ad"

        missing = [p for p in (sc_path, st_path) if not p.exists()]
        if missing:
            msg = f"[Pair {pair_id:>3}] Missing files: {[str(p) for p in missing]}"
            logger.error(msg)
            errors.append(msg)
            continue

        keys = cell_type_keys(sc_name)
        if not keys:
            msg = (
                f"[Pair {pair_id:>3}] No CellTypeKey found in index for "
                f"'{sc_name}' - skipping"
            )
            logger.warning(msg)
            errors.append(msg)
            continue

        pair_dir = output_dir / f"{pair_id:03d}_{sc_name}__{st_name}"

        for ct_key in keys:
            granularity_dir = pair_dir / ct_key
            mapping_prob_path = granularity_dir / "mapping_prob.h5ad"
            tag = f"[Pair {pair_id:>3} | {ct_key}]"

            logger.info("%s Running %s: %s x %s", tag, aligner, sc_name, st_name)
            try:
                run_aligner(aligner, sc_path, st_path, granularity_dir, ct_key)
            except Exception as exc:
                msg = f"{tag} {aligner} FAILED: {exc}"
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

            logger.info("%s Running mapping analysis", tag)
            try:
                analyze_mapping(sc_path, st_path, granularity_dir, cell_type_key=ct_key)
            except Exception as exc:
                msg = f"{tag} mapping analysis FAILED: {exc}"
                logger.error(msg)
                errors.append(msg)

        logger.info("[Pair %3d] Done -> %s", pair_id, pair_dir)

    if errors:
        logger.warning("%d error(s) occurred:", len(errors))
        for e in errors:
            logger.warning("  %s", e)
    else:
        logger.info("All pairs completed successfully.")
    return errors
