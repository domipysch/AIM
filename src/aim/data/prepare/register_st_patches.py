"""
Register the output of split_st_into_patches.py into the database.

split_st_into_patches.py writes <base>_patches_NxN/patch_<r>_<c>/st.h5ad but does
NOT touch the database. This script copies those patch files into ST/ under the
slice naming convention {nn}_{i}_{tissue}.h5ad (as used by datasets 05/09/10) and
replaces the monolithic ST entry with one row per patch in ST/index.csv and pairs.csv.

Patches are numbered 1..N in row-major order over the non-empty patches. The
monolithic scRNA reference is reused for every patch (scName unchanged). The
monolithic ST h5ad is left on disk but removed from index.csv/pairs.csv (delete it
yourself once you are happy with the split).

Usage:
  python -m data_preparation.register_st_patches \
      --patches-dir "C:/Users/zi69hebi/Dev/10_Alignment/Data/01_Datasets/ST/11_human-breast-cancer_patches_5x5" \
      [--data-root "C:/Users/zi69hebi/Dev/10_Alignment/Data/01_Datasets"] \
      [--dry-run]
"""

import argparse
import csv
import logging
import re
import shutil
import sys
from pathlib import Path

import anndata as ad

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(r"/01_Datasets")
_PATCH_RE = re.compile(r"^patch_(\d+)_(\d+)$")


def _st_sort_key(name: str) -> tuple[int, int, int]:
    m = re.match(r"^(\d+)_(\d+)_(\d+)_", name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.match(r"^(\d+)_(\d+)_", name)
    if m:
        return int(m.group(1)), int(m.group(2)), 0
    m = re.match(r"^(\d+)_", name)
    if m:
        return int(m.group(1)), 0, 0
    return 999, 999, 999


def _ordered_patches(patches_dir: Path) -> list[tuple[Path, int]]:
    """Return [(patch_st_h5ad, n_spots), ...] in row-major (r, c) order, non-empty only."""
    dirs = []
    for d in patches_dir.glob("patch_*"):
        m = _PATCH_RE.match(d.name)
        if m and (d / "st.h5ad").exists():
            dirs.append((int(m.group(1)), int(m.group(2)), d / "st.h5ad"))
    dirs.sort()
    out = []
    for _, _, h5 in dirs:
        n = ad.read_h5ad(h5).n_obs
        out.append((h5, n))
    return out


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--patches-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    st_dir = args.data_root / "ST"
    st_index_path = st_dir / "index.csv"
    pairs_path = args.data_root / "pairs.csv"

    # Base ST name = patches dir name minus the "_patches_NxN" suffix
    base_name = re.sub(r"_patches_\d+x\d+$", "", args.patches_dir.name)
    m = re.match(r"^(\d+)_(.+)$", base_name)
    if not m:
        logger.error("Cannot parse dataset id from base name %r", base_name)
        sys.exit(1)
    dataset_id, tissue = m.group(1), m.group(2)
    logger.info(
        "Base ST dataset: %s  (id=%s, tissue=%s)", base_name, dataset_id, tissue
    )

    patches = _ordered_patches(args.patches_dir)
    if not patches:
        logger.error(
            "No non-empty patch st.h5ad files found under %s", args.patches_dir
        )
        sys.exit(1)
    total = sum(n for _, n in patches)
    logger.info("%d non-empty patches, %d spots total", len(patches), total)

    # --- Load DB tables ---
    with open(st_index_path, newline="") as f:
        st_reader = csv.DictReader(f)
        st_fields = list(st_reader.fieldnames or [])
        st_rows = list(st_reader)
    with open(pairs_path, newline="") as f:
        pairs_reader = csv.DictReader(f)
        pairs_fields = list(pairs_reader.fieldnames or [])
        pairs_rows = list(pairs_reader)

    st_template = next((r for r in st_rows if r.get("Name") == base_name), None)
    if st_template is None:
        logger.error("ST/index.csv has no row named %r", base_name)
        sys.exit(1)
    base_pairs = [r for r in pairs_rows if r.get("stName") == base_name]
    if not base_pairs:
        logger.error("pairs.csv has no pair with stName %r", base_name)
        sys.exit(1)

    # --- Build new slice rows / pairs / file copies ---
    new_st_rows, new_pairs, copies = [], [], []
    for i, (h5, n_spots) in enumerate(patches, start=1):
        new_name = f"{dataset_id}_{i}_{tissue}"
        copies.append((h5, st_dir / f"{new_name}.h5ad"))
        new_st_rows.append({**st_template, "Name": new_name, "SpotCount": str(n_spots)})
        for bp in base_pairs:
            new_pairs.append({**bp, "stName": new_name})

    # Replace monolith with slices
    st_rows = [r for r in st_rows if r.get("Name") != base_name] + new_st_rows
    pairs_rows = [r for r in pairs_rows if r.get("stName") != base_name] + new_pairs
    st_rows.sort(key=lambda r: _st_sort_key(r["Name"]))
    for new_id, r in enumerate(pairs_rows):
        r["PairID"] = str(new_id)

    if args.dry_run:
        logger.info("[dry-run] would copy %d files, e.g.:", len(copies))
        for src, dst in copies[:3]:
            logger.info("    %s  ->  %s", src, dst.name)
        logger.info("[dry-run] ST/index.csv: -1 monolith, +%d slices", len(new_st_rows))
        logger.info(
            "[dry-run] pairs.csv: -%d monolith pair(s), +%d, renumber to 0..%d",
            len(base_pairs),
            len(new_pairs),
            len(pairs_rows) - 1,
        )
        return

    for src, dst in copies:
        shutil.copyfile(src, dst)
    logger.info("Copied %d patch files into %s", len(copies), st_dir)

    shutil.copyfile(st_index_path, st_index_path.with_suffix(".csv.bak"))
    with open(st_index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=st_fields)
        w.writeheader()
        w.writerows(st_rows)
    logger.info("Updated %s (backup .bak): %d rows", st_index_path.name, len(st_rows))

    shutil.copyfile(pairs_path, pairs_path.with_suffix(".csv.bak"))
    with open(pairs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pairs_fields)
        w.writeheader()
        w.writerows(pairs_rows)
    logger.info("Updated %s (backup .bak): %d pairs", pairs_path.name, len(pairs_rows))

    logger.info(
        "Monolith %s.h5ad left on disk (now unregistered); delete it when ready.",
        base_name,
    )
    logger.info(
        "Done. Validate with: aim validate --pairs_csv %s",
        Path(args.data_root) / "pairs.csv",
    )


if __name__ == "__main__":
    main()
