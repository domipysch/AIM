from pathlib import Path

import anndata
import tacco as tc
import pandas as pd
import numpy as np
import logging
from anndata import AnnData
import argparse


def tacco_map(
    adata_sc: AnnData,
    adata_st: AnnData,
    map_cell_types: bool,
    cell_type_key: str,
    output_folder: Path,
):
    """
    Run TACCO alignment on already-loaded AnnData objects and write
    ``mapping_prob.h5ad`` to ``output_folder``.

    The inputs are treated as read-only (copied before ``tc.tl.annotate``, which
    mutates ``adata_st``) so they can be reused across many K by
    :class:`AlignerWorker`.

    Args:
        adata_sc: Single-cell reference (C x G).
        adata_st: Spatial data (S x G).
        map_cell_types: If True, aggregate cells by cell_type_key before mapping.
                        If False, map individual cells directly.
        cell_type_key: obs column to use as annotation when map_cell_types=True.
        output_folder: Folder where to store mapping_prob.h5ad.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Copy up front — tc.tl.annotate writes into adata_st (and cell mode adds an
    # obs column to adata_sc), and the worker reuses the same objects per K.
    adata_sc = adata_sc.copy()
    adata_st = adata_st.copy()

    # Determine which obs column to use as the annotation key for TACCO
    if map_cell_types:
        if cell_type_key not in adata_sc.obs.columns:
            raise KeyError(
                f"cell_type_key '{cell_type_key}' not found in obs columns "
                f"{list(adata_sc.obs.columns)}."
            )
        annotation_col = cell_type_key
    else:
        # Map individual cells: expose obs_names (cellID) as a column
        annotation_col = adata_sc.obs.index.name or "cellID"
        adata_sc.obs[annotation_col] = adata_sc.obs_names.tolist()

    # Map with TACCO
    logging.info("Align data with TACCO (annotation_col=%s)", annotation_col)
    try:
        tc.tl.annotate(
            adata_st,
            adata_sc,
            annotation_key=annotation_col,
            result_key="align_result",
            remove_constant_genes=False,  # We have issues with some datasets without this argument
        )
    except ValueError as e:
        if "observations without non-zero variables" not in str(e):
            raise
        # TACCO's bisection boost can zero out spots mid-run; retry without bisection
        logging.warning("TACCO bisection failed (%s) — retrying with bisections=0", e)
        tc.tl.annotate(
            adata_st,
            adata_sc,
            annotation_key=annotation_col,
            result_key="align_result",
            remove_constant_genes=False,
            bisections=0,
        )
    # Mapping now in adata_st.obsm["align_result"]

    # Fractions from TACCO (ensure row sums ~1), S x T
    # TACCO already returns this as a DataFrame indexed by spot with columns named
    # after the actual cell type / cell values from annotation_col.
    fractions_prob = adata_st.obsm["align_result"]
    logging.info("Shape fractions: %s", fractions_prob.shape)

    # Save mapping as AnnData (obs=spots, var=cell types)
    ad_map = AnnData(
        X=fractions_prob.values.astype(np.float32),
        obs=pd.DataFrame(index=fractions_prob.index),
        var=pd.DataFrame(index=fractions_prob.columns.astype(str)),
    )
    ad_map.write_h5ad(output_folder / "mapping_prob.h5ad")
    logging.info("Saved mapping to %s", output_folder / "mapping_prob.h5ad")


def _load_pair(sc_path: Path, st_path: Path) -> tuple[AnnData, AnnData]:
    assert Path(sc_path).exists(), f"sc.h5ad not found: {sc_path}"
    assert Path(st_path).exists(), f"st.h5ad not found: {st_path}"
    logging.info("Load data")
    adata_sc = anndata.read_h5ad(Path(sc_path))  # C x G
    adata_st = anndata.read_h5ad(Path(st_path))  # S x G
    return adata_sc, adata_st


def tacco_align_data(
    sc_path: Path,
    st_path: Path,
    map_cell_types: bool,
    cell_type_key: str,
    output_folder: Path,
):
    """One-shot: load the pair from disk and run :func:`tacco_map` once."""
    adata_sc, adata_st = _load_pair(sc_path, st_path)
    tacco_map(
        adata_sc,
        adata_st,
        map_cell_types=map_cell_types,
        cell_type_key=cell_type_key,
        output_folder=output_folder,
    )


def _serve(sc_path: Path, st_path: Path, control_dir: Path) -> None:
    """Load the pair once, then map one K per job (canonical baseline settings)."""
    from reference_aligners.registry import serve_loop

    adata_sc, adata_st = _load_pair(sc_path, st_path)

    def handle_job(cell_type_key: str, output_folder: Path) -> None:
        tacco_map(
            adata_sc,
            adata_st,
            map_cell_types=True,
            cell_type_key=cell_type_key,
            output_folder=output_folder,
        )

    serve_loop(control_dir, handle_job)


if __name__ == "__main__":
    """
    Run TACCO alignment on a prepared dataset at given folder.
    Settings can be modified in the code below.
    """

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run TACCO alignment on a dataset folder"
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
        help="Folder where to store results (required unless --server)",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        required=False,
        default="cellType",
        help="What cell type key to load from sc data as cell type annotation to be mapped.",
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
        tacco_align_data(
            args.scdata,
            args.stdata,
            map_cell_types=args.cell_type_key is not None,
            cell_type_key=args.cell_type_key,
            output_folder=args.output_folder,
        )
