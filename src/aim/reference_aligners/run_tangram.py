import os
from pathlib import Path

import anndata
import tangram as tg
import pandas as pd
import scanpy as sc
import numpy as np
import logging
import argparse
import torch
from anndata import AnnData

logger = logging.getLogger(__name__)


def tangram_map(
    adata_sc: AnnData,
    adata_st: AnnData,
    normalize_and_log: bool,
    compute_marker_genes: bool,
    map_clusters: bool,
    cell_type_key: str,
    output_folder: Path,
):
    """
    Run Tangram alignment on already-loaded AnnData objects and write
    ``mapping_prob.h5ad`` to ``output_folder``.

    The inputs are treated as read-only: ``adata_sc``/``adata_st`` are copied
    before any in-place normalization / Tangram pre-processing, so the same
    objects can be reused across many K by :class:`AlignerWorker`.

    Args:
        adata_sc: Single-cell reference (C x G).
        adata_st: Spatial data (S x G).
        normalize_and_log: Should the sc and st input data be normalized and log-transformed before alignment?
        compute_marker_genes: Whether to compute marker genes (as proposed in Tangram Tutorials) or use all genes.
        map_clusters: Whether to use cluster-based mapping (cell types) or cell-based mapping.
        cell_type_key: What cell type key to load from sc data as cell type annotation.
        output_folder: Folder where to store results to.
    """
    output_folder = Path(output_folder)
    # Copy up front — normalize_total / log1p / tg.pp_adatas mutate in place, and
    # the worker hands the same adata_st in for every K.
    adata_sc = adata_sc.copy()
    adata_st = adata_st.copy()

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info("Map cells to spots with Tangram")
    if map_clusters:
        ad_map_prob = tg.map_cells_to_space(
            adata_sc_map,
            adata_st,
            mode="clusters",
            cluster_label=cell_type_key,
            density_prior="rna_count_based",
            num_epochs=500,
            device=device,
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
            device=device,
            lambda_g1=1,
            # lambda_g2=1,
        )  # C x S
        assert ad_map_prob.n_obs == adata_sc_map.n_obs
        assert ad_map_prob.n_vars == adata_st.n_obs

    # Step 4: Reshape mapping to spots x type, with proper cell type names in var_names
    # (Tangram's own obs_names are just numeric indices for cluster mode; the actual
    # cell type names live in the cell_type_key obs column instead)
    if map_clusters:
        type_names = ad_map_prob.obs[cell_type_key].astype(str).tolist()
    else:
        type_names = ad_map_prob.obs_names.tolist()

    ad_map = AnnData(
        X=ad_map_prob.X.T.astype(np.float32),
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=type_names),
    )

    mapping_path_h5ad = output_folder / "mapping_prob.h5ad"
    ad_map.write_h5ad(mapping_path_h5ad)
    logger.info("Saved mapping to %s", mapping_path_h5ad)


def _load_pair(sc_path: Path, st_path: Path) -> tuple[AnnData, AnnData]:
    assert Path(sc_path).exists(), f"sc.h5ad not found: {sc_path}"
    assert Path(st_path).exists(), f"st.h5ad not found: {st_path}"
    logger.info("Load data")
    adata_sc = anndata.read_h5ad(Path(sc_path))  # C x G
    adata_st = anndata.read_h5ad(Path(st_path))  # S x G
    logger.info(f"Single Cell Data: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    logger.info(f"Spatial Data: {adata_st.n_obs} spots x {adata_st.n_vars} genes")
    return adata_sc, adata_st


def tangram_align_data(
    sc_path: Path,
    st_path: Path,
    normalize_and_log: bool,
    compute_marker_genes: bool,
    map_clusters: bool,
    cell_type_key: str,
    output_folder: Path,
):
    """One-shot: load the pair from disk and run :func:`tangram_map` once."""
    adata_sc, adata_st = _load_pair(sc_path, st_path)
    tangram_map(
        adata_sc,
        adata_st,
        normalize_and_log=normalize_and_log,
        compute_marker_genes=compute_marker_genes,
        map_clusters=map_clusters,
        cell_type_key=cell_type_key,
        output_folder=output_folder,
    )


def _serve(sc_path: Path, st_path: Path, control_dir: Path) -> None:
    """Load the pair once, then map one K per job (canonical baseline settings)."""
    from aim.reference_aligners.registry import serve_loop

    adata_sc, adata_st = _load_pair(sc_path, st_path)

    def handle_job(cell_type_key: str, output_folder: Path) -> None:
        tangram_map(
            adata_sc,
            adata_st,
            normalize_and_log=False,
            compute_marker_genes=False,
            map_clusters=True,
            cell_type_key=cell_type_key,
            output_folder=output_folder,
        )

    serve_loop(control_dir, handle_job)


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
        "--output_folder",
        type=Path,
        help="Path where to store result (required unless --server)",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        required=False,
        default="cellType",
        help="What cell type key to load from sc data as cell type annotation to be mapped.",
    )
    parser.add_argument(
        "-nal",
        "--normalize_and_log",
        action="store_true",
        required=False,
        default=False,
        help="Whether to normalize and log input data beforehand",
    )
    parser.add_argument(
        "--compute_marker_genes",
        action="store_true",
        default=False,
        help="Whether to compute marker genes before mapping",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Persistent-worker mode: load the pair once and map one K per job "
        "from --control_dir (see reference_aligners.registry.AlignerWorker).",
    )
    parser.add_argument(
        "--control_dir",
        type=Path,
        default=None,
        help="Job-queue directory (required with --server)",
    )

    args = parser.parse_args()

    if args.server:
        if args.control_dir is None:
            parser.error("--server requires --control_dir")
        _serve(args.scdata, args.stdata, args.control_dir)
    else:
        if args.output_folder is None:
            parser.error("--output_folder is required")
        tangram_align_data(
            args.scdata,
            args.stdata,
            normalize_and_log=args.normalize_and_log,
            compute_marker_genes=args.compute_marker_genes,
            map_clusters=args.cell_type_key is not None,
            cell_type_key=args.cell_type_key,
            output_folder=args.output_folder,
        )
