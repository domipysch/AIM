"""
Ingest the 10x Genomics / Janesick et al. 2023 human breast cancer dataset into the
database as a new scRNA <-> ST pair (dataset id 11).

Sources (as downloaded, see README "Datensaetze"):
  scRNA reference  scFFPE-seq (Chromium Flex), GSM7782698 raw feature-barcode matrix
                   + cell-type annotations from the scFFPE-Seq sheet of
                   GSE243275_Barcode_Cell_Type_Matrices.xlsx
  ST (high-res)    Xenium In Situ (313-plex), <outs>/cell_feature_matrix.h5
                   + cell centroids from <outs>/cells.parquet
                   + supervised Xenium labels from the "Xenium R1 Fig1-5 (supervised)" sheet

Output (matching the database h5ad contract, see CLAUDE.md):
  scRNA/11_human-breast-cancer.h5ad   X=raw counts (annotated cells x genes),
                                       obs['cellType'], var uppercase + unique
  ST/11_human-breast-cancer.h5ad      X=raw counts (313 panel), obsm['spatial'],
                                       obs['cellType'] (Xenium ground truth)
and appends one row to scRNA/index.csv, ST/index.csv and pairs.csv (originals backed
up to *.bak). ExperimentMatch = same_paper.

Usage:
  python -m data_preparation.build_xenium_breast_pair \
      --xenium-outs   "C:/Users/zi69hebi/Downloads/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/outs" \
      --scffpe-h5     "C:/Users/zi69hebi/Downloads/GSE243275/GSM7782698_count_raw_feature_bc_matrix.h5" \
      --annotation-xlsx "C:/Users/zi69hebi/Downloads/GSE243275_Barcode_Cell_Type_Matrices.xlsx" \
      [--data-root "C:/Users/zi69hebi/Dev/10_Alignment/Data/01_Datasets"] \
      [--dry-run]
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logger = logging.getLogger(__name__)

DATASET_ID = "11"
NAME = "11_human-breast-cancer"
TISSUE = "Human breast cancer"
QUELLE = "https://www.nature.com/articles/s41467-023-43458-x"
AUTOR = "Janesick et al"
FEATURED_IN = "DOT"
SC_TECH = "Chromium Flex (scFFPE-seq)"
ST_TECH = "Xenium"
SC_LINK = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243275"
ST_LINK = (
    "https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast"
)
EXPERIMENT_MATCH = "same_paper"

SCFFPE_SHEET = "scFFPE-Seq"
XENIUM_SHEET = "Xenium R1 Fig1-5 (supervised)"

DEFAULT_DATA_ROOT = Path(r"C:/Users/zi69hebi/Dev/10_Alignment/Data/01_Datasets")


def _uppercase_and_collapse(adata: ad.AnnData) -> ad.AnnData:
    """Uppercase var_names, then sum counts of any genes that collide into one column.

    Keeps var_names unique + uppercase so validate_database passes and shared-gene
    intersection with the ST panel is exact.
    """
    names = pd.Index([g.upper() for g in adata.var_names])
    if names.is_unique:
        adata.var_names = names
        return adata

    uniq = names.unique()
    col_of = {g: i for i, g in enumerate(uniq)}
    cols = np.fromiter((col_of[g] for g in names), dtype=np.int64, count=len(names))
    # (n_old x n_new) 0/1 grouping matrix; X @ G sums duplicate columns
    group = sp.csr_matrix(
        (np.ones(len(names), dtype=np.float32), (np.arange(len(names)), cols)),
        shape=(len(names), len(uniq)),
    )
    x_new = sp.csr_matrix(adata.X).dot(group)
    n_collapsed = len(names) - len(uniq)
    logger.info(
        "  collapsed %d duplicate gene symbol(s) by summing counts", n_collapsed
    )
    out = ad.AnnData(
        X=x_new.astype(np.float32),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=uniq),
    )
    return out


def build_scrna(scffpe_h5: Path, annotation_xlsx: Path) -> ad.AnnData:
    logger.info("scRNA: reading raw matrix %s", scffpe_h5.name)
    adata = sc.read_10x_h5(scffpe_h5)
    logger.info("  raw matrix: %d barcodes x %d genes", adata.n_obs, adata.n_vars)

    ann = pd.read_excel(annotation_xlsx, sheet_name=SCFFPE_SHEET)
    ann = ann.set_index("Barcode")["Annotation"]
    logger.info(
        "  annotation sheet '%s': %d cells, %d types",
        SCFFPE_SHEET,
        len(ann),
        ann.nunique(),
    )

    keep = adata.obs_names.intersection(ann.index)
    missing = len(ann) - len(keep)
    if missing:
        logger.warning("  %d annotated barcode(s) not found in raw matrix", missing)
    adata = adata[keep].copy()
    adata.obs["cellType"] = ann.reindex(adata.obs_names).astype(str).values
    logger.info("  subset to %d annotated cells", adata.n_obs)

    # raw integer counts as float32 sparse
    adata.X = sp.csr_matrix(adata.X).astype(np.float32)
    adata = _uppercase_and_collapse(adata)
    adata.obs_names = [str(b) for b in adata.obs_names]
    adata.obs_names_make_unique()
    logger.info("  final scRNA: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def build_st(xenium_outs: Path, annotation_xlsx: Path) -> ad.AnnData:
    h5 = xenium_outs / "cell_feature_matrix.h5"
    logger.info("ST: reading Xenium matrix %s", h5.name)
    adata = sc.read_10x_h5(h5, gex_only=False)
    if "feature_types" in adata.var:
        is_gene = adata.var["feature_types"].astype(str) == "Gene Expression"
        adata = adata[:, is_gene.values].copy()
    logger.info("  kept %d Gene Expression features", adata.n_vars)

    cells = pd.read_parquet(xenium_outs / "cells.parquet")
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id")
    coords = cells.reindex(adata.obs_names)[["x_centroid", "y_centroid"]]
    if coords.isnull().any().any():
        raise RuntimeError("ST: some cell barcodes have no centroid in cells.parquet")
    adata.obsm["spatial"] = coords.to_numpy(dtype=np.float32)

    # Xenium supervised ground-truth labels (bonus obs column; not required by schema)
    xann = pd.read_excel(annotation_xlsx, sheet_name=XENIUM_SHEET)
    xann = xann.set_index(xann.columns[0])[xann.columns[1]]
    xann.index = xann.index.astype(str)
    adata.obs["cellType"] = xann.reindex(adata.obs_names).astype(str).values

    adata.X = sp.csr_matrix(adata.X).astype(np.float32)
    adata = _uppercase_and_collapse(adata)
    adata.obs_names = [str(b) for b in adata.obs_names]
    adata.obs_names_make_unique()
    logger.info("  final ST: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def _append_row(csv_path: Path, row: dict, dry_run: bool) -> None:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    name = row.get("Name")
    sc_n, st_n = row.get("scName"), row.get("stName")
    if name and "Name" in fields:
        duplicate = any(r.get("Name") == str(name) for r in rows)
    elif sc_n and st_n:
        duplicate = any(
            r.get("scName") == str(sc_n) and r.get("stName") == str(st_n) for r in rows
        )
    else:
        duplicate = False
    if duplicate:
        logger.warning(
            "  %s already contains this entry - not appending again", csv_path.name
        )
        return

    full = {k: str(row.get(k, "")) for k in fields}
    unknown = set(row) - set(fields)
    if unknown:
        logger.warning(
            "  %s: ignoring columns not in header: %s", csv_path.name, unknown
        )

    if dry_run:
        logger.info("  [dry-run] would append to %s: %s", csv_path.name, full)
        return

    shutil.copyfile(csv_path, csv_path.with_suffix(".csv.bak"))
    # Guard against a source file that lacks a trailing newline, which would
    # otherwise glue the new row onto the last existing row.
    with open(csv_path, "rb") as f:
        needs_newline = f.seek(0, 2) > 0 and (f.seek(-1, 2), f.read(1))[1] not in (
            b"\n",
            b"\r",
        )
    with open(csv_path, "a", newline="") as f:
        if needs_newline:
            f.write("\r\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(full)
    logger.info(
        "  appended row to %s (backup: %s)",
        csv_path.name,
        csv_path.with_suffix(".csv.bak").name,
    )


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--xenium-outs", type=Path, required=True)
    p.add_argument("--scffpe-h5", type=Path, required=True)
    p.add_argument("--annotation-xlsx", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sc_dir = args.data_root / "scRNA"
    st_dir = args.data_root / "ST"
    sc_out = sc_dir / f"{NAME}.h5ad"
    st_out = st_dir / f"{NAME}.h5ad"

    adata_sc = build_scrna(args.scffpe_h5, args.annotation_xlsx)
    adata_st = build_st(args.xenium_outs, args.annotation_xlsx)

    shared = adata_sc.var_names.intersection(adata_st.var_names)
    logger.info("shared genes (sc-int-st): %d", len(shared))
    n_types = int(adata_sc.obs["cellType"].nunique())

    if not args.dry_run:
        adata_sc.write_h5ad(sc_out)
        logger.info("wrote %s", sc_out)
        adata_st.write_h5ad(st_out)
        logger.info("wrote %s", st_out)
    else:
        logger.info("[dry-run] would write %s and %s", sc_out, st_out)

    _append_row(
        sc_dir / "index.csv",
        {
            "Name": NAME,
            "Tissue": TISSUE,
            "Quelle": QUELLE,
            "Autor": AUTOR,
            "Tech": SC_TECH,
            "FeaturedIn": FEATURED_IN,
            "CellCount": adata_sc.n_obs,
            "GeneCount": adata_sc.n_vars,
            "CellTypeKey0": "cellType",
            "NumberCellTypes0": n_types,
            "Links": SC_LINK,
        },
        args.dry_run,
    )

    _append_row(
        st_dir / "index.csv",
        {
            "Name": NAME,
            "Tissue": TISSUE,
            "Quelle": QUELLE,
            "Autor": AUTOR,
            "Tech": ST_TECH,
            "FeaturedIn": FEATURED_IN,
            "SpotCount": adata_st.n_obs,
            "GeneCount": adata_st.n_vars,
            "Links": ST_LINK,
        },
        args.dry_run,
    )

    pairs_path = args.data_root / "pairs.csv"
    with open(pairs_path, newline="") as f:
        existing = list(csv.DictReader(f))
    next_id = (
        max(
            (int(r["PairID"]) for r in existing if r.get("PairID", "").isdigit()),
            default=-1,
        )
        + 1
    )
    _append_row(
        pairs_path,
        {
            "PairID": next_id,
            "scName": NAME,
            "stName": NAME,
            "NumberSharedGenes": len(shared),
            "ExperimentMatch": EXPERIMENT_MATCH,
        },
        args.dry_run,
    )

    logger.info(
        "Done. Next: python -m data_preparation.validate_database -d %s", args.data_root
    )


if __name__ == "__main__":
    main()
