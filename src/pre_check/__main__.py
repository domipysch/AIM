import argparse
import logging
from pathlib import Path

import scanpy as sc

from . import run_pre_check

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser(
    description="Pre-alignment compatibility check for SC/ST dataset pairs."
)
parser.add_argument("--scdata", required=True, type=Path)
parser.add_argument("--stdata", required=True, type=Path)
parser.add_argument("--output_dir", required=True, type=Path)
parser.add_argument("--leiden_resolution", default=1.0, type=float)
parser.add_argument("--permutation_test", action="store_true")
parser.add_argument("--n_permutations", default=200, type=int)
args = parser.parse_args()

sc_adata = sc.read_h5ad(args.scdata)
st_adata = sc.read_h5ad(args.stdata)

run_pre_check(
    sc_adata=sc_adata,
    st_adata=st_adata,
    output_dir=args.output_dir,
    leiden_resolution=args.leiden_resolution,
    run_permutation_test=args.permutation_test,
    n_permutations=args.n_permutations,
)
