# Takes a st.h5ad file and selects the top N highly variable genes,
# then saves the filtered AnnData to a new h5ad file.
#
# Usage:
#   python subset_st_hvg.py <path_to_st.h5ad> <N>
# Output:
#   <path_to_st>_top<N>hvg.h5ad  (same directory as input)

import sys
import os
import anndata as ad
import scanpy as sc


def subset_st_to_top_hvg(input_path: str, n: int) -> None:
    stData = ad.read_h5ad(input_path)
    print(f"Loaded ST data: {stData.n_obs} spots x {stData.n_vars} genes")

    stLog = stData.copy()
    sc.pp.normalize_total(stLog)
    sc.pp.log1p(stLog)
    sc.pp.highly_variable_genes(stLog, n_top_genes=n, inplace=True)

    hvgs = stLog.var_names[stLog.var["highly_variable"].values].tolist()
    if len(hvgs) == 0:
        raise RuntimeError("No HVGs selected — check input data and N value")

    stSubset = stData[:, hvgs].copy()

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_top{n}hvg{ext}"
    stSubset.write_h5ad(output_path)
    print(f"Saved: {output_path}  ({stSubset.n_obs} spots x {stSubset.n_vars} genes)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python subset_st_hvg.py <path_to_st.h5ad> <N>")
        sys.exit(1)

    input_path = sys.argv[1]
    n = int(sys.argv[2])
    subset_st_to_top_hvg(input_path, n)
