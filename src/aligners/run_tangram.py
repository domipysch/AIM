import os
from pathlib import Path
import tangram as tg
import pandas as pd
import scanpy as sc
import numpy as np
import logging
from anndata import AnnData
import argparse
from scipy.sparse import issparse
from ..utils.io import load_sc_adata, load_st_adata

logger = logging.getLogger(__name__)


def tangram_align_data(
    sc_path: Path,
    st_path: Path,
    normalize_and_log: bool,
    compute_marker_genes: bool,
    map_clusters: bool,
    cell_type_key: str,
    output_folder: Path,
) -> tuple[AnnData, AnnData]:
    """
    Run Tangram alignment on a prepared dataset.
    Saves prediction GEP CSV to output_path.

    Args:
        sc_path: Full path to sc.h5ad
        st_path: Full path to st.h5ad
        normalize_and_log: Should the sc and st input data be normalized and log-transformed before alignment?
        compute_marker_genes: Whether to compute marker genes (as proposed in Tangram Tutorials) or use all genes.
        map_clusters: Whether to use cluster-based mapping (cell types) or cell-based mapping.
        cell_type_key: What cell type key to load from sc data as cell type annotation.
        output_folder: Folder where to store results to.

    Returns: -
    """
    assert Path(sc_path).exists(), f"sc.h5ad not found: {sc_path}"
    assert Path(st_path).exists(), f"st.h5ad not found: {st_path}"

    # Load input scRNA and ST data
    logger.info("Load data")
    adata_sc = load_sc_adata(Path(sc_path))  # C x G
    adata_st = load_st_adata(Path(st_path))  # S x G
    logger.info("Data loaded")
    logger.info(f"Single Cell Data: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    logger.info(f"Spatial Data: {adata_st.n_obs} spots x {adata_st.n_vars} genes")

    # Create directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Step 1 (optional): Compute marker genes (optional, speeds up mapping)
    if compute_marker_genes:
        logger.info("Define marker genes")
        adata_sc_copy = adata_sc.copy()

        # Filter out cell types with only one cell for marker gene computation
        singletons = (
            adata_sc_copy.obs[cell_type_key]
            .value_counts()
            .loc[lambda x: x == 1]
            .index.tolist()
        )
        adata_sc_copy = adata_sc_copy[
            ~adata_sc_copy.obs[cell_type_key].isin(singletons)
        ].copy()

        # Proposed in Tangram tutorials: normalize & log transform first
        sc.pp.normalize_total(adata_sc_copy)
        adata_sc_copy.X = np.log1p(adata_sc_copy.X)

        # Create list of names of marker genes
        sc.tl.rank_genes_groups(adata_sc_copy, groupby=cell_type_key, use_raw=False)
        markers_df = pd.DataFrame(adata_sc_copy.uns["rank_genes_groups"]["names"]).iloc[
            0:100, :
        ]
        markers: list[str] = list(np.unique(markers_df.melt().value.values))

    adata_sc_map = adata_sc.copy()
    if normalize_and_log:
        # Normalize & log-transform input data (optional, as in Tangram tutorials)
        logger.info("Normalize & Log-transform gene expression and spatial data")
        sc.pp.normalize_total(adata_sc_map)
        sc.pp.normalize_total(adata_st)
        adata_sc_map.X = np.log1p(adata_sc_map.X)
        adata_st.X = np.log1p(adata_st.X)

    # Step 2: Tangram pre-processing
    # (see https://github.com/broadinstitute/Tangram/blob/master/tangram/mapping_utils.py)
    if compute_marker_genes:
        logger.info(f"Pre-process data with Tangram with {len(markers)} marker genes")
        tg.pp_adatas(adata_sc_map, adata_st, genes=markers)
    else:
        logger.info(f"Pre-process data with Tangram with all genes as marker genes")
        tg.pp_adatas(adata_sc_map, adata_st, genes=None)

    # Step 3: Mapping
    logger.info("Map cells to spots with Tangram")
    if map_clusters:
        ad_map_prob = tg.map_cells_to_space(
            adata_sc_map,
            adata_st,
            mode="clusters",
            cluster_label=cell_type_key,
            density_prior="rna_count_based",
            num_epochs=500,
            device="cpu",
        )  # T x S
        assert ad_map_prob.n_obs == len(adata_sc_map.obs[cell_type_key].unique())
        assert ad_map_prob.n_vars == adata_st.n_obs
    else:
        ad_map_prob = tg.map_cells_to_space(
            adata_sc_map,
            adata_st,
            mode="cells",
            density_prior="rna_count_based",
            num_epochs=500,
            device="cpu",
            lambda_g1=1,
            # lambda_g2=1,
        )  # C x S
        assert ad_map_prob.n_obs == adata_sc_map.n_obs
        assert ad_map_prob.n_vars == adata_st.n_obs

    # Step 4: Make coppy of mapping: Apply one-hot encoding to mapping: Only one cell / cell type per spot
    ad_map_det = ad_map_prob.copy()

    # For each column (spot) in ad_map, set the max value to 1 and all others to 0 (map spot to exactly one cell)
    if issparse(ad_map_det.X):
        mat = ad_map_det.X.toarray()
    else:
        mat = ad_map_det.X.copy()
    argmax_idx = np.argmax(
        mat, axis=0
    )  # for each spot (column), index of max cell / cell type
    one_hot = np.zeros_like(mat, dtype=float)
    one_hot[argmax_idx, np.arange(mat.shape[1])] = 1.0
    ad_map_det.X = one_hot

    for ad_map, mapping_type in ((ad_map_prob, "prob"), (ad_map_det, "det")):

        # Store mapping matrix as h5ad
        mapping_path_h5ad = output_folder / f"mapping_{mapping_type}.h5ad"
        ad_map.write_h5ad(mapping_path_h5ad)
        logger.info(f"Saved {mapping_type} mapping to %s", mapping_path_h5ad)

        # Compute Z' out of the mapping (expected gene expression per spot, scRNA data weighted by mapping)
        logger.info("Project gene expression to spatial spots")
        ad_ge = tg.project_genes(
            adata_map=ad_map,
            adata_sc=adata_sc,
            cluster_label=cell_type_key if map_clusters else None,
        )  # S x G

        # Transpose to G x S
        ad_ge = ad_ge.transpose()  # now G x S
        assert ad_ge.n_obs == adata_sc.n_vars
        assert ad_ge.n_vars == adata_st.n_obs

        # Export ad_ge to CSV
        # - Rows: Genes
        # - Columns: Spots
        # - Top left cell = "GEP"
        if issparse(ad_ge.X):
            expr = ad_ge.X.A
        else:
            expr = ad_ge.X

        # Check: Rows = Genes, Columns = Spots
        assert expr.shape == (adata_sc.n_vars, adata_st.n_obs), "dims passen nicht"

        # Make obs_names uppercase (gene names in uppercase in input data, also in output)
        ad_ge.obs_names = [s.upper() for s in ad_ge.obs_names]

        # Write resulting GEP h5ad (layout: obs = genes G, var = spots S)
        logger.info(f"Write result GEP to h5ad.")
        gep_path_h5ad = output_folder / f"gep_{mapping_type}.h5ad"
        ad_ge.write_h5ad(gep_path_h5ad)
        logger.info(f"Saved {mapping_type} tangram GEP to {gep_path_h5ad}")

    return ad_ge


if __name__ == "__main__":
    """
    Run Tangram alignment on a prepared dataset at given folder.
    Settings can be modified in the code below.
    """
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run Tangram alignment on a dataset folder"
    )
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "-o", "--output_folder", type=Path, help="Path where to store result"
    )
    parser.add_argument(
        "-nal",
        "--normalize_and_log",
        action="store_true",
        help="Whether to normalize and log input data beforehand",
    )
    parser.add_argument(
        "--compute_marker_genes",
        action="store_true",
        default=False,
        help="Whether to compute marker genes before mapping",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        required=False,
        default="cellType",
        help="What cell type key to load from sc data as cell type annotation to be mapped. If not provided, no cell type annotation is loaded and mapping is done cell-based.",
    )
    args = parser.parse_args()

    tangram_align_data(
        args.scdata,
        args.stdata,
        normalize_and_log=args.normalize_and_log,
        compute_marker_genes=args.compute_marker_genes,
        map_clusters=args.cell_type_key is not None,
        cell_type_key=args.cell_type_key,
        output_folder=args.output_folder,
    )
