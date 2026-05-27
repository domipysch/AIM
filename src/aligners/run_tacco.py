from pathlib import Path
import tacco as tc
import pandas as pd
import numpy as np
import logging
from anndata import AnnData
import argparse
from scipy.sparse import issparse
from ..utils.io import load_sc_adata, load_st_adata


def tacco_align_data(
    sc_path: Path,
    st_path: Path,
    map_cell_types: bool,
    cell_type_key: str,
    output_folder: Path,
) -> tuple[AnnData, AnnData]:
    """
    Run TACCO alignment on a prepared dataset.
    Saves both probabilistic and deterministic GEPs to output_folder.

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        map_cell_types: If True, aggregate cells by cell_type_key before mapping.
                        If False, map individual cells directly.
        cell_type_key: obs column to use as annotation when map_cell_types=True.
        output_folder: Folder where to store gep_prob.h5ad and gep_det.h5ad.

    Returns:
        Tuple (gep_prob, gep_det), each AnnData with obs=genes, var=spots (G x S layout).
    """
    assert Path(sc_path).exists(), f"sc.h5ad not found: {sc_path}"
    assert Path(st_path).exists(), f"st.h5ad not found: {st_path}"

    output_folder.mkdir(parents=True, exist_ok=True)

    logging.info("Load data")
    adata_sc = load_sc_adata(Path(sc_path))  # C x G
    adata_st = load_st_adata(Path(st_path))  # S x G

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
            assume_valid_counts=True,
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
            assume_valid_counts=True,
            remove_constant_genes=False,
            bisections=0,
        )
    # Mapping now in adata_st.obsm["align_result"]

    # Compute mean expression per annotation group (cells x genes -> groups x genes)
    if issparse(adata_sc.X):
        Xsc = adata_sc.X.toarray()
    else:
        Xsc = np.array(adata_sc.X)
    sc_obs = pd.DataFrame(
        Xsc, index=adata_sc.obs_names, columns=adata_sc.var_names
    )  # C x G
    mean_expr = sc_obs.groupby(adata_sc.obs[annotation_col]).mean()  # T x G

    # Convert mean -> gene profile p_tg (rows sum to 1)
    p = mean_expr.values
    p_sum = p.sum(axis=1, keepdims=True)
    p_sum[p_sum == 0] = 1.0  # avoid division by zero for empty types
    p_tg = p / p_sum  # T x G
    logging.info("Shape p_tg: %s", p_tg.shape)

    # Fractions from TACCO (ensure row sums ~1), S x T
    fractions_prob = pd.DataFrame(
        adata_st.obsm["align_result"], index=adata_st.obs_names, columns=mean_expr.index
    )
    logging.info("Shape fractions: %s", fractions_prob.shape)

    # Deterministic fractions: argmax one-hot per spot
    idxmax_series = fractions_prob.idxmax(axis=1)
    col_idx = fractions_prob.columns.get_indexer(idxmax_series)
    one_hot_vals = np.zeros_like(fractions_prob.values, dtype=np.uint8)
    one_hot_vals[np.arange(len(fractions_prob)), col_idx] = 1
    fractions_det = pd.DataFrame(
        one_hot_vals,
        index=fractions_prob.index,
        columns=fractions_prob.columns,
        dtype=np.float32,
    )

    # Total counts per spot
    if issparse(adata_st.X):
        spot_counts = np.array(adata_st.X.sum(axis=1)).flatten()
    else:
        spot_counts = np.array(adata_st.X.sum(axis=1)).flatten()

    def _build_gep(fractions: pd.DataFrame) -> AnnData:
        C_st = fractions.to_numpy() * spot_counts[:, None]  # S x T
        recon = C_st @ p_tg  # S x G
        assert recon.shape == (adata_st.n_obs, adata_sc.n_vars), "dims passen nicht"
        adata_recon = AnnData(
            X=recon.T.astype(np.float32),
            obs=pd.DataFrame(index=adata_sc.var_names),
            var=pd.DataFrame(index=adata_st.obs_names),
        )
        adata_recon.obs_names = list(adata_sc.var_names)
        adata_recon.var_names = list(adata_st.obs_names)
        return adata_recon

    gep_prob = _build_gep(fractions_prob)
    gep_det = _build_gep(fractions_det)

    for fractions, label in ((fractions_prob, "prob"), (fractions_det, "det")):
        # Save mapping as AnnData (obs=cell_types, var=spots) — same layout as Tangram
        ad_map = AnnData(
            X=fractions.values.T.astype(np.float32),
            obs=pd.DataFrame(index=fractions.columns),
            var=pd.DataFrame(index=fractions.index),
        )
        ad_map.write_h5ad(output_folder / f"mapping_{label}.h5ad")
        logging.info(
            "Saved %s mapping to %s", label, output_folder / f"mapping_{label}.h5ad"
        )

    gep_prob.write_h5ad(output_folder / "gep_prob.h5ad")
    logging.info("Saved prob GEP to %s", output_folder / "gep_prob.h5ad")

    gep_det.write_h5ad(output_folder / "gep_det.h5ad")
    logging.info("Saved det GEP to %s", output_folder / "gep_det.h5ad")

    return gep_prob, gep_det


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
        "-o",
        "--output_folder",
        type=Path,
        required=True,
        help="Folder where to store results",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        required=False,
        default="cellType",
        help="What cell type key to load from sc data as cell type annotation to be mapped. If not provided, no cell type annotation is loaded and mapping is done cell-based.",
    )
    args = parser.parse_args()

    tacco_align_data(
        args.scdata,
        args.stdata,
        map_cell_types=args.cell_type_key is not None,
        cell_type_key=args.cell_type_key,
        output_folder=args.output_folder,
    )
