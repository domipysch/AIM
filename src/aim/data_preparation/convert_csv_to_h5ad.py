"""
Convert a dataset from the legacy 6-CSV format to two h5ad files.

Legacy format (per dataset folder):
  scData_Cells.csv  — cell metadata (cellID, cellType, cellTypeMinor, ...)
  scData_Genes.csv  — gene IDs (one per row, convenience only)
  scData_GEP.csv    — expression matrix (rows: genes, cols: cells)
  stData_Spots.csv  — spot metadata (spotID, cArray0, cArray1)
  stData_Genes.csv  — gene IDs (one per row, convenience only)
  stData_GEP.csv    — expression matrix (rows: genes, cols: spots)

New format:
  sc.h5ad  — AnnData (C x G): X = expression, obs = cell metadata, var = gene IDs
  st.h5ad  — AnnData (S x G): X = expression, obs = spot metadata,
             var = gene IDs, obsm["spatial"] = (S, 2) float array

Usage:
  # Convert a single dataset folder:
  python -m src.data_preparation.convert_csv_to_h5ad -d <dataset_folder>

  # Convert every subdirectory under a root folder:
  python -m src.data_preparation.convert_csv_to_h5ad -d <root_folder> --all

  # Also compute PCA + UMAP embeddings for cellxgene compatibility:
  python -m src.data_preparation.convert_csv_to_h5ad -d <dataset_folder> --compute-embeddings
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

logger = logging.getLogger(__name__)


def _compute_embeddings(adata_sc: ad.AnnData) -> None:
    """
    Compute PCA and UMAP embeddings on a normalized copy and store them in adata_sc.
    Operates on a copy to avoid modifying the raw counts in adata_sc.X.
    Stores results as obsm["X_pca"] and obsm["X_umap"].
    """
    import scanpy as sc

    logger.info("  Computing embeddings (normalize → log1p → PCA → UMAP)...")
    tmp = adata_sc.copy()
    sc.pp.normalize_total(tmp)
    sc.pp.log1p(tmp)
    n_pcs = min(50, tmp.n_obs - 1, tmp.n_vars - 1)
    sc.pp.pca(tmp, n_comps=n_pcs)
    sc.pp.neighbors(tmp, n_pcs=n_pcs)
    sc.tl.umap(tmp)
    adata_sc.obsm["X_pca"] = tmp.obsm["X_pca"]
    adata_sc.obsm["X_umap"] = tmp.obsm["X_umap"]
    logger.info("  Embeddings stored (X_pca, X_umap)")


def convert_dataset(
    folder: Path, overwrite: bool = False, compute_embeddings: bool = False
) -> None:
    """
    Convert one dataset folder from CSV to h5ad.

    Args:
        folder: Path to the dataset folder containing the 6 CSV files.
        overwrite: If False, skip folders where sc.h5ad and st.h5ad already exist.
        compute_embeddings: If True, compute PCA + UMAP and store in sc.h5ad obsm.
    """
    sc_out = folder / "sc.h5ad"
    st_out = folder / "st.h5ad"

    if not overwrite and sc_out.exists() and st_out.exists():
        logger.info("Skipping %s (h5ad files already exist)", folder.name)
        return

    logger.info("Converting %s ...", folder.name)

    # --- scRNA ---
    sc_gep_path = folder / "scData_GEP.csv"
    sc_cells_path = folder / "scData_Cells.csv"

    if not sc_gep_path.exists():
        raise FileNotFoundError(f"Missing scData_GEP.csv in {folder}")
    if not sc_cells_path.exists():
        raise FileNotFoundError(f"Missing scData_Cells.csv in {folder}")

    # GEP is G x C on disk; transpose to C x G
    sc_gep_df = pd.read_csv(sc_gep_path, index_col=0)  # G x C
    sc_cells_df = pd.read_csv(sc_cells_path)

    # Build obs: use cellID as index (if column exists), else use GEP column names
    if "cellID" in sc_cells_df.columns:
        sc_cells_df = sc_cells_df.set_index("cellID")
    else:
        sc_cells_df.index = sc_gep_df.columns

    adata_sc = ad.AnnData(
        X=sc_gep_df.T.values.astype(np.float32),
        obs=sc_cells_df.reindex(sc_gep_df.columns),
        var=pd.DataFrame(index=sc_gep_df.index),
    )
    adata_sc.obs_names = list(sc_gep_df.columns)
    adata_sc.var_names = list(sc_gep_df.index)

    if compute_embeddings:
        _compute_embeddings(adata_sc)

    adata_sc.write_h5ad(sc_out)
    logger.info("  Written %s", sc_out)

    # --- Spatial ---
    st_gep_path = folder / "stData_GEP.csv"
    st_spots_path = folder / "stData_Spots.csv"

    if not st_gep_path.exists():
        raise FileNotFoundError(f"Missing stData_GEP.csv in {folder}")
    if not st_spots_path.exists():
        raise FileNotFoundError(f"Missing stData_Spots.csv in {folder}")

    # GEP is G x S on disk; transpose to S x G
    st_gep_df = pd.read_csv(st_gep_path, index_col=0)  # G x S
    st_spots_df = pd.read_csv(st_spots_path)

    # Build obs: use spotID as index if available
    if "spotID" in st_spots_df.columns:
        st_spots_df = st_spots_df.set_index("spotID")
    else:
        st_spots_df.index = st_gep_df.columns

    coords = st_spots_df[["cArray0", "cArray1"]].values.astype(np.float32)
    obs_meta = st_spots_df.drop(columns=["cArray0", "cArray1"], errors="ignore")

    adata_st = ad.AnnData(
        X=st_gep_df.T.values.astype(np.float32),
        obs=obs_meta.reindex(st_gep_df.columns),
        var=pd.DataFrame(index=st_gep_df.index),
    )
    adata_st.obs_names = list(st_gep_df.columns)
    adata_st.var_names = list(st_gep_df.index)
    adata_st.obsm["spatial"] = coords

    adata_st.write_h5ad(st_out)
    logger.info("  Written %s", st_out)


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert dataset(s) from 6-CSV format to sc.h5ad + st.h5ad"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        required=True,
        help="Path to a single dataset folder, or (with --all) the root folder containing dataset subfolders",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every immediate subdirectory of -d",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing h5ad files (default: skip if both already exist)",
    )
    parser.add_argument(
        "--compute-embeddings",
        action="store_true",
        help="Compute PCA + UMAP embeddings and store in sc.h5ad (required for cellxgene)",
    )
    args = parser.parse_args()

    if args.all:
        root = args.dataset
        if not root.is_dir():
            print(f"ERROR: {root} is not a directory", file=sys.stderr)
            sys.exit(1)
        folders = sorted(p for p in root.iterdir() if p.is_dir())
        if not folders:
            print(f"No subdirectories found under {root}", file=sys.stderr)
            sys.exit(1)
    else:
        folders = [args.dataset]

    errors = []
    for folder in folders:
        try:
            convert_dataset(
                folder,
                overwrite=args.overwrite,
                compute_embeddings=args.compute_embeddings,
            )
        except Exception as e:
            logger.error("Failed to convert %s: %s", folder.name, e)
            errors.append((folder.name, e))

    if errors:
        print(f"\n{len(errors)} dataset(s) failed:", file=sys.stderr)
        for name, exc in errors:
            print(f"  {name}: {exc}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll conversions completed successfully.")


if __name__ == "__main__":
    main()
