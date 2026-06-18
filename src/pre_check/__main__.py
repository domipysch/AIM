import argparse
import logging
from pathlib import Path
import scanpy as sc
from . import run_pre_check

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Pre-alignment compatibility check for SC/ST dataset pairs."
    )
    parser.add_argument(
        "--sc", required=True, type=Path, help="Path to scRNA-seq .h5ad file."
    )
    parser.add_argument(
        "--st",
        required=True,
        type=Path,
        help="Path to spatial transcriptomics .h5ad file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for the pre-check report.",
    )
    parser.add_argument("--leiden_resolution", required=False, type=float, default=0.5)
    args = parser.parse_args()

    run_pre_check(
        sc_adata=sc.read_h5ad(args.sc),
        st_adata=sc.read_h5ad(args.st),
        output_dir=args.output,
        leiden_resolution=args.leiden_resolution,
    )
