"""
Ingest three imaging-based spatial <-> scRNA pairs harvested from the Li et al. 2022
Nature Methods benchmark into the database:

  ob     dataset 12  mouse olfactory bulb  seqFISH+ (Eng 2019)  x  Drop-seq (GSE148360)
  mtg    dataset 13  human middle temporal gyrus  ISS (SpaceTx)  x  Allen Human MTG Smart-seq (Hodge 2019)
  cortex dataset 14  mouse cortex  seqFISH+ (Eng 2019)  x  Allen Mouse V1/ALM Smart-seq (Tasic 2018)

Produces scRNA/<name>.h5ad + ST/<name>.h5ad (database h5ad contract: raw counts,
uppercase unique var, obs['cellType'] on scRNA, obsm['spatial'] on ST) and appends a
row to scRNA/index.csv, ST/index.csv and pairs.csv (originals backed up to *.bak).

seqFISH/seqFISH+ spatial data is tiled into multiple fields of view with per-FOV local
coordinates; we grid-offset the FOVs so the section is one non-overlapping spatial map
and keep the FOV id in obs so it can be re-split later.

Usage:
  python -m data_preparation.build_imaging_pairs --pair ob     --dl-dir <scratch/dl> [--data-root ...] [--dry-run]
  python -m data_preparation.build_imaging_pairs --pair mtg    --dl-dir <scratch/dl> [...]
  python -m data_preparation.build_imaging_pairs --pair cortex --dl-dir <scratch/dl> [...]
"""

import argparse
import csv
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from aim.data_preparation.build_xenium_breast_pair import (
    _uppercase_and_collapse,
    _append_row,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(r"C:/Users/zi69hebi/Dev/10_Alignment/Data/01_Datasets")

META = {
    "ob": dict(
        did="12",
        name="12_mouse-olfactory-bulb",
        tissue="Mouse olfactory bulb",
        sc_autor="Brann et al",
        sc_tech="Drop-seq",
        sc_quelle="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148360",
        sc_link="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148360",
        st_autor="Eng et al",
        st_tech="seqFISH+",
        st_quelle="https://www.nature.com/articles/s41586-019-1049-y",
        st_link="https://github.com/CaiGroup/seqFISH-PLUS",
        featured="Li2022",
        match="-",
    ),
    "mtg": dict(
        did="12",
        name="12_human-mtg",
        tissue="Human middle temporal gyrus",
        sc_autor="Hodge et al",
        sc_tech="Smart-seq",
        sc_quelle="https://www.nature.com/articles/s41586-019-1506-7",
        sc_link="https://portal.brain-map.org/atlases-and-data/rnaseq/human-mtg-smart-seq",
        st_autor="SpaceTx consortium",
        st_tech="ISS",
        st_quelle="https://www.nature.com/articles/s41592-022-01480-9",
        st_link="https://github.com/spacetx-spacejam/data",
        featured="Li2022",
        match="same_consortium",
    ),
    "cortex": dict(
        did="14",
        name="14_mouse-cortex",
        tissue="Mouse cortex (VISp/ALM)",
        sc_autor="Tasic et al",
        sc_tech="Smart-seq",
        sc_quelle="https://www.nature.com/articles/s41586-018-0654-5",
        sc_link="https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-v1-and-alm-smart-seq",
        st_autor="Eng et al",
        st_tech="seqFISH+",
        st_quelle="https://www.nature.com/articles/s41586-019-1049-y",
        st_link="https://github.com/CaiGroup/seqFISH-PLUS",
        featured="Li2022",
        match="-",
    ),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _grid_offset_fov(coords: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Lay per-FOV local coordinates onto a padded grid so tiles do not overlap."""
    coords = coords.astype(float)
    fovs = np.unique(fov)
    ncol = int(np.ceil(np.sqrt(len(fovs))))
    tile_w = (coords[:, 0].max() - coords[:, 0].min()) * 1.1
    tile_h = (coords[:, 1].max() - coords[:, 1].min()) * 1.1
    out = coords.copy()
    for k, f in enumerate(fovs):
        r, c = divmod(k, ncol)
        m = fov == f
        out[m, 0] = coords[m, 0] - coords[m, 0].min() + c * tile_w
        out[m, 1] = coords[m, 1] - coords[m, 1].min() + r * tile_h
    return out.astype(np.float32)


def _finalize_sc(adata: ad.AnnData) -> ad.AnnData:
    adata.X = sp.csr_matrix(adata.X).astype(np.float32)
    adata = _uppercase_and_collapse(adata)
    adata.obs_names = [str(b) for b in adata.obs_names]
    adata.obs_names_make_unique()
    return adata


def _load_allen(zip_paths: list[Path], region_tags: list[str]) -> ad.AnnData:
    """Load Allen Smart-seq region zip(s): exon+intron summed, cells x gene-symbols,
    obs['cellType']=cluster. Detects whether the matrix row key is symbol or entrez.

    Memory-safe: extracts the 4 members to disk, then streams exon+intron in gene
    chunks summing into one preallocated int32 array (avoids pandas int64 blow-up)."""
    parts = []
    for zp, tag in zip(zip_paths, region_tags):
        logger.info("  reading Allen zip %s", zp.name)
        tmp = zp.parent / f"allen_tmp_{tag}"
        tmp.mkdir(exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            pick = lambda suf: next(n for n in names if n.endswith(suf))  # noqa: E731
            for suf in (
                "genes-rows.csv",
                "samples-columns.csv",
                "exon-matrix.csv",
                "intron-matrix.csv",
            ):
                out = tmp / suf
                if not out.exists() or out.stat().st_size == 0:
                    with z.open(pick(suf)) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 24)

        grows = pd.read_csv(tmp / "genes-rows.csv")
        scols = pd.read_csv(tmp / "samples-columns.csv")
        cells = pd.read_csv(
            tmp / "exon-matrix.csv", index_col=0, nrows=0
        ).columns.to_numpy()
        n_genes, n_cells = len(grows), len(cells)
        X = np.empty((n_genes, n_cells), dtype=np.int32)
        gene_key = np.empty(n_genes, dtype=object)
        pos = 0
        ex_it = pd.read_csv(
            tmp / "exon-matrix.csv", index_col=0, dtype=np.int32, chunksize=4000
        )
        in_it = pd.read_csv(
            tmp / "intron-matrix.csv", index_col=0, dtype=np.int32, chunksize=4000
        )
        for ce, ci in zip(ex_it, in_it):
            k = ce.shape[0]
            X[pos : pos + k] = ce.to_numpy() + ci.to_numpy()
            gene_key[pos : pos + k] = ce.index.to_numpy().astype(str)
            pos += k
        X, gene_key = X[:pos], gene_key[:pos]
        shutil.rmtree(tmp, ignore_errors=True)

        # map matrix row key -> gene symbol
        sym_col = "gene_symbol" if "gene_symbol" in grows.columns else "gene"
        best, key_col = -1, None
        for c in grows.columns:
            ov = np.isin(gene_key, grows[c].astype(str).to_numpy()).mean()
            if ov > best:
                best, key_col = ov, c
        logger.info(
            "  gene key column=%s (match=%.2f) -> symbol column=%s",
            key_col,
            best,
            sym_col,
        )
        id2sym = dict(zip(grows[key_col].astype(str), grows[sym_col].astype(str)))
        var_syms = [id2sym.get(str(g), str(g)) for g in gene_key]

        scols = scols.set_index(scols.columns[0])
        clusters = (
            scols.reindex([str(c) for c in cells])["cluster"].astype(str).to_numpy()
        )

        a = ad.AnnData(
            X=sp.csr_matrix(X.T).astype(np.float32),
            obs=pd.DataFrame(
                {"cellType": clusters, "region": tag},
                index=[f"{tag}_{c}" for c in cells],
            ),
            var=pd.DataFrame(index=var_syms),
        )
        parts.append(a)

    if len(parts) > 1:
        adata = ad.concat(parts, join="outer")
        adata.X = sp.csr_matrix(np.nan_to_num(adata.X.toarray())).astype(np.float32)
    else:
        adata = parts[0]
    # drop cells without a real cluster label
    bad = {"nan", "NA", "", "none", "low quality", "Low Quality"}
    keep = ~pd.Series(adata.obs["cellType"].astype(str).to_numpy()).isin(bad).to_numpy()
    adata = adata[keep].copy()
    return adata


# --------------------------------------------------------------------------- #
# per-pair builders
# --------------------------------------------------------------------------- #
def build_ob(dl: Path) -> tuple[ad.AnnData, ad.AnnData]:
    sd = dl / "seqfish" / "sourcedata"
    counts = pd.read_csv(sd / "ob_counts.csv")  # cells x genes
    cen = pd.read_csv(sd / "ob_cellcentroids.csv")  # FoV, Cell ID, X, Y
    ann = pd.read_csv(
        dl / "seqfish_ann" / "OB_cell_type_annotations.csv"
    )  # index, louvain
    st = ad.AnnData(
        X=sp.csr_matrix(counts.to_numpy(dtype=np.float32)),
        var=pd.DataFrame(index=counts.columns.astype(str)),
    )
    st.obs_names = [f"ob_{i}" for i in range(st.n_obs)]
    st.obs["fov"] = cen["Field of View"].to_numpy()
    st.obs["louvain"] = ann["louvain"].to_numpy()
    st.obsm["spatial"] = _grid_offset_fov(
        cen[["X", "Y"]].to_numpy(), cen["Field of View"].to_numpy()
    )
    st = _uppercase_and_collapse(st)

    logger.info("  reading OB scRNA (genes x cells)")
    gep = pd.read_csv(dl / "ob_scrna_counts.csv.gz", index_col=0)  # genes x cells
    meta = pd.read_csv(dl / "meta_datta.csv.gz", index_col=0)  # barcode -> Cell_class
    keep = [b for b in gep.columns if b in meta.index]
    sc = ad.AnnData(
        X=sp.csr_matrix(gep[keep].T.to_numpy(dtype=np.float32)),
        var=pd.DataFrame(index=gep.index.astype(str)),
    )
    sc.obs_names = keep
    sc.obs["cellType"] = meta.reindex(keep)["Cell_class"].astype(str).to_numpy()
    sc.obs["louvain"] = meta.reindex(keep)["seurat_clusters"].astype(str).to_numpy()
    sc = _finalize_sc(sc)
    return sc, st


def build_cortex(dl: Path) -> tuple[ad.AnnData, ad.AnnData]:
    sd = dl / "seqfish" / "sourcedata"
    counts = pd.read_csv(sd / "cortex_svz_counts.csv")
    cen = pd.read_csv(sd / "cortex_svz_cellcentroids.csv")
    ann = pd.read_csv(dl / "seqfish_ann" / "cortex_svz_cell_type_annotations.csv")
    mask = (cen["Region"] == "Cortex").to_numpy()
    st = ad.AnnData(
        X=sp.csr_matrix(counts.to_numpy(dtype=np.float32)[mask]),
        var=pd.DataFrame(index=counts.columns.astype(str)),
    )
    cen, louv = cen[mask].reset_index(drop=True), ann["louvain"].to_numpy()[mask]
    st.obs_names = [f"cortex_{i}" for i in range(st.n_obs)]
    st.obs["fov"] = cen["Field of View"].to_numpy()
    st.obs["louvain"] = louv
    st.obsm["spatial"] = _grid_offset_fov(
        cen[["X", "Y"]].to_numpy(), cen["Field of View"].to_numpy()
    )
    st = _uppercase_and_collapse(st)

    sc = (
        _load_allen([dl / "dl_visp", dl / "dl_alm"], ["VISp", "ALM"])
        if (dl / "dl_visp").exists()
        else _load_allen([dl / "visp.zip", dl / "alm.zip"], ["VISp", "ALM"])
    )
    sc = _finalize_sc(sc)
    return sc, st


def build_mtg(dl: Path) -> tuple[ad.AnnData, ad.AnnData]:
    df = pd.read_csv(dl / "iss_mtg_540.csv")  # CellID, genes..., centroidX/Y
    gene_cols = [c for c in df.columns if c not in ("CellID", "centroidX", "centroidY")]
    st = ad.AnnData(
        X=sp.csr_matrix(df[gene_cols].to_numpy(dtype=np.float32)),
        var=pd.DataFrame(index=[g.upper() for g in gene_cols]),
    )
    st.obs_names = [f"iss_{c}" for c in df["CellID"].to_numpy()]
    st.obsm["spatial"] = df[["centroidX", "centroidY"]].to_numpy(dtype=np.float32)
    st = _uppercase_and_collapse(st)

    sc = _load_allen([dl / "human_mtg.zip"], ["MTG"])
    sc = _finalize_sc(sc)
    return sc, st


BUILDERS = {"ob": build_ob, "mtg": build_mtg, "cortex": build_cortex}


# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pair", choices=list(BUILDERS), required=True)
    p.add_argument(
        "--dl-dir", type=Path, required=True, help="scratch dir with downloaded inputs"
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    m = META[args.pair]
    sc, st = BUILDERS[args.pair](args.dl_dir)
    shared = sc.var_names.intersection(st.var_names)
    n_types = int(sc.obs["cellType"].nunique())
    logger.info(
        "%s: sc=%d x %d (%d types), st=%d x %d, shared=%d",
        m["name"],
        sc.n_obs,
        sc.n_vars,
        n_types,
        st.n_obs,
        st.n_vars,
        len(shared),
    )

    sc_out = args.data_root / "scRNA" / f"{m['name']}.h5ad"
    st_out = args.data_root / "ST" / f"{m['name']}.h5ad"
    if args.dry_run:
        logger.info("[dry-run] would write %s and %s", sc_out, st_out)
        return

    sc.write_h5ad(sc_out)
    logger.info("wrote %s", sc_out)
    st.write_h5ad(st_out)
    logger.info("wrote %s", st_out)

    _append_row(
        args.data_root / "scRNA" / "index.csv",
        {
            "Name": m["name"],
            "Tissue": m["tissue"],
            "Quelle": m["sc_quelle"],
            "Autor": m["sc_autor"],
            "Tech": m["sc_tech"],
            "FeaturedIn": m["featured"],
            "CellCount": sc.n_obs,
            "GeneCount": sc.n_vars,
            "CellTypeKey0": "cellType",
            "NumberCellTypes0": n_types,
            "Links": m["sc_link"],
        },
        False,
    )
    _append_row(
        args.data_root / "ST" / "index.csv",
        {
            "Name": m["name"],
            "Tissue": m["tissue"],
            "Quelle": m["st_quelle"],
            "Autor": m["st_autor"],
            "Tech": m["st_tech"],
            "FeaturedIn": m["featured"],
            "SpotCount": st.n_obs,
            "GeneCount": st.n_vars,
            "Links": m["st_link"],
        },
        False,
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
            "scName": m["name"],
            "stName": m["name"],
            "NumberSharedGenes": len(shared),
            "ExperimentMatch": m["match"],
        },
        False,
    )
    logger.info("Done %s.", m["name"])


if __name__ == "__main__":
    main()
