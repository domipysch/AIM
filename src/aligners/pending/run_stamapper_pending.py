import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from anndata import AnnData
import argparse
from scipy.sparse import issparse

from STAMapper import pipeline
from STAMapper.utils.train import seed_everything
import anndata

logger = logging.getLogger(__name__)

_DATASET_NAMES = ("scRNA", "spatial")


def stamapper_align_data(
    sc_path: Path,
    st_path: Path,
    normalize_and_log: bool,
    deterministic_mapping: bool,
    cell_type_key: str,
    output_path: Path,
    n_epochs: int = 350,
    seed: int = 42,
) -> AnnData:
    """
    Run STAMapper alignment on a prepared dataset.
    Saves a probability-mapping h5ad and a projected GEP h5ad.

    STAMapper always operates at cell-type level (no single-cell mapping mode),
    so there is no map_clusters argument.  The model trains a GNN with dropout
    and is therefore stochastic; fix `seed` for reproducibility.

    `deterministic_mapping` applies an argmax (one-hot) to the per-spot
    cell-type probability matrix before projecting gene expression, analogous
    to Tangram's deterministic mode.  The underlying model probabilities are
    always soft — this flag only affects how the GEP is projected.

    Args:
        sc_path:               Full path to sc.h5ad
        st_path:               Full path to st.h5ad
        normalize_and_log:     Normalize and log-transform input data before alignment.
        deterministic_mapping: Apply argmax to the probability matrix for GEP projection.
        cell_type_key:         obs column with cell-type labels in sc data.
        output_path:           Full path where to save the GEP h5ad.
        n_epochs:              Training epochs (200-400 recommended for whole-graph).
        seed:                  Random seed for reproducibility.

    Returns:
        AnnData with obs = genes, var = spots (GEP, same layout as run_tangram output).
    """
    assert Path(sc_path).exists(), f"sc.h5ad not found: {sc_path}"
    assert Path(st_path).exists(), f"st.h5ad not found: {st_path}"

    output_path = Path(output_path).with_suffix(".h5ad")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)

    logger.info("Load data")
    adata_sc = anndata.readh5ad(Path(sc_path))
    adata_st = anndata.readh5ad(Path(st_path))
    logger.info(
        "Single Cell Data: %d cells x %d genes", adata_sc.n_obs, adata_sc.n_vars
    )
    logger.info(
        "Spatial Data:      %d spots x %d genes", adata_st.n_obs, adata_st.n_vars
    )

    # pipeline.training() writes checkpoints to ./_temp/{dsnames}/0/ relative to CWD.
    # Change to the output directory so those files land alongside our results.
    original_cwd = os.getcwd()
    os.chdir(output_path.parent)
    try:
        logger.info("Train STAMapper model")
        outputs = pipeline.training(
            adatas=[adata_sc, adata_st],
            dsnames=_DATASET_NAMES,
            key_classes=[cell_type_key, None],
        )
    finally:
        os.chdir(original_cwd)

    # --- extract probability matrix for spatial spots only ---
    dpair = outputs["dpair"]
    df_probs_all = outputs["df_probs"]  # (n_sc + n_spatial) x n_celltypes

    spatial_mask = dpair.obs["dataset"] == _DATASET_NAMES[1]
    spatial_indices = np.where(spatial_mask.values)[0]
    df_probs_spatial = df_probs_all.iloc[spatial_indices].copy()
    df_probs_spatial.index = adata_st.obs_names  # S x T

    cell_types = list(df_probs_spatial.columns)

    # --- save mapping h5ad (T x S), matching Tangram's cluster-mode convention ---
    probs_T_S = df_probs_spatial.values.T  # T x S
    if deterministic_mapping:
        argmax_idx = np.argmax(probs_T_S, axis=0)
        one_hot = np.zeros_like(probs_T_S, dtype=float)
        one_hot[argmax_idx, np.arange(probs_T_S.shape[1])] = 1.0
        probs_T_S = one_hot

    ad_map = AnnData(
        X=probs_T_S,
        obs=pd.DataFrame(index=cell_types),
        var=pd.DataFrame(index=adata_st.obs_names),
    )
    mapping_path = output_path.with_name(
        output_path.stem.replace("_GEP", "_mapping") + ".h5ad"
    )
    ad_map.write_h5ad(mapping_path)
    logger.info("Saved mapping to %s", mapping_path)

    # --- project gene expression: GEP[spot, gene] = sum_t(p[spot,t] * mean_expr[t,gene]) ---
    logger.info("Project gene expression to spatial spots")
    ct_mean_rows = []
    for ct in cell_types:
        mask = adata_sc.obs[cell_type_key] == ct
        expr = adata_sc[mask].X
        if issparse(expr):
            expr = expr.toarray()
        ct_mean_rows.append(expr.mean(axis=0))
    mean_expr = np.vstack(ct_mean_rows)  # T x G

    df_probs_for_gep = pd.DataFrame(
        probs_T_S.T, index=adata_st.obs_names, columns=cell_types
    )
    gep_S_G = df_probs_for_gep.values @ mean_expr  # S x G

    ad_ge = AnnData(
        X=gep_S_G,
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=adata_sc.var_names),
    )
    ad_ge = ad_ge.T  # -> G x S  (obs=genes, var=spots)
    ad_ge.obs_names = [s.upper() for s in ad_ge.obs_names]

    assert ad_ge.n_obs == adata_sc.n_vars
    assert ad_ge.n_vars == adata_st.n_obs

    logger.info("Write result GEP to h5ad")
    ad_ge.write_h5ad(output_path)
    logger.info("Saved STAMapper GEP to %s", output_path)

    return ad_ge


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run STAMapper alignment on a dataset folder"
    )
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path where to store result GEP h5ad",
    )
    parser.add_argument(
        "-nal",
        "--normalize_and_log",
        action="store_true",
        help="Normalize and log-transform input data beforehand",
    )
    parser.add_argument(
        "-det",
        "--deterministic",
        action="store_true",
        help="Apply argmax to probability matrix before GEP projection",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        default="cellType",
        help="Cell type obs column in sc data (default: cellType)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=350,
        help="Number of training epochs (default: 350)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    stamapper_align_data(
        args.scdata,
        args.stdata,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=args.deterministic,
        cell_type_key=args.cell_type_key,
        output_path=args.output_path,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
