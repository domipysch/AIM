"""
Uppercases var_names in all h5ad files under scRNA/ and ST/ subdirectories.
Run once after adding new datasets to ensure consistent gene name casing.

Usage:
    python -m src.data_preparation.normalize_gene_names
    python -m src.data_preparation.normalize_gene_names -d /path/to/01_Datasets
"""

import argparse
from pathlib import Path
import anndata as ad


def normalize_file(path: Path) -> None:
    adata = ad.read_h5ad(path)
    original = adata.var_names.tolist()
    uppercased = [g.upper() for g in original]

    if original == uppercased:
        print(f"  {path.name}: already uppercase, skipped")
        return

    changed = sum(1 for a, b in zip(original, uppercased) if a != b)
    adata.var_names = uppercased
    adata.write_h5ad(path)
    print(f"  {path.name}: uppercased {changed} gene names")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uppercase var_names in all h5ad files"
    )
    parser.add_argument(
        "--data-root",
        "-d",
        type=Path,
        default=Path(r"/01_Datasets"),
    )
    args = parser.parse_args()

    root: Path = args.data_root
    for subdir in ["scRNA", "ST"]:
        folder = root / subdir
        h5ads = sorted(folder.glob("*.h5ad"))
        if not h5ads:
            continue
        print(f"\n{subdir}/")
        for path in h5ads:
            normalize_file(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
