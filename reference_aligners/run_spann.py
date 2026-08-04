"""Run SPANN as a reference aligner.

SPANN (https://github.com/ddb-qiwang/SPANN-torch, Bib 2024 bbad533) annotates
single-cell-resolution spatial data from an annotated scRNA reference via a coupled
VAE + minibatch unbalanced optimal transport. Unlike Tangram/TACCO/DOT it produces a
single hard cell-type label per spatial cell (``pred_cls``) plus a novel-cell
"E-score". To keep the output comparable to the other baselines we disable novel-cell
detection (``novel_cell_test=False``, ``resolution=1.0`` -> "no novel cells"), so every
spatial cell is annotated with one of the reference cell types.

The output contract matches the other aligners: ``output_folder/mapping_prob.h5ad`` is
an AnnData with ``obs`` = spots (aligned to the ST spot order), ``var`` = cell-type
names, ``X`` = float32 (S x T). Because SPANN only yields a hard label, the soft
assignment is the one-hot of ``pred_cls`` (metrics row-normalise internally; argmax is
unaffected -- the same representation the in-process ``nearest_centroid`` mapper produces).
"""

# SPANN pins Python 3.9 (see environment_spann.yml), which evaluates function
# annotations eagerly and so rejects PEP 604 ``str | None`` unions at import time.
# Defer all annotations to strings so the modern syntax is accepted.
from __future__ import annotations

from pathlib import Path

import argparse
import logging

import anndata
import numpy as np
import pandas as pd
import torch
from anndata import AnnData

# SPANN's pip package exposes these at spann.preprocess / spann.model. generate_ae_params
# lives in preprocess (the tutorial reaches it via a star-import from model, but its
# definition is in preprocess.py).
from spann.preprocess import (
    anndata_preprocess,
    generate_dataloaders,
    generate_ae_params,
)
from spann.model import SPANN_model

logger = logging.getLogger(__name__)


def spann_align_data(
    sc_path: Path,
    st_path: Path,
    cell_type_key: str,
    output_folder: Path,
    device: str | None = None,
    highly_variable: int = 2000,
    batch_size: int = 256,
    feat_dim: int = 16,
    lr: float = 2e-4,
    lambda_recon: float = 2000,
    lambda_kl: float = 0.5,
    lambda_spa: float = 0.1,
    lambda_cd: float = 0.001,
    lambda_nb: float = 0.1,
    maxiter: int = 6000,
    miditer1: int = 2000,
    miditer2: int = 5000,
    miditer3: int = 4000,
) -> None:
    """Run SPANN alignment on a prepared sc/st pair and write ``mapping_prob.h5ad``.

    Args:
        sc_path: Full path to sc.h5ad (raw counts; ``cell_type_key`` in ``obs``).
        st_path: Full path to st.h5ad (raw counts; spatial coords in ``obsm['spatial']``
            or ``obs['X']``/``obs['Y']``).
        cell_type_key: obs column on the reference to use as the cell-type annotation.
        output_folder: Folder where ``mapping_prob.h5ad`` is written.
        device: Torch device string ("cuda"/"cpu"); auto-detected when None.
        highly_variable/batch_size/feat_dim/lr/lambda_*/*iter: SPANN hyperparameters
            (SPANN defaults; expose so batch runs can tune them). Novel-cell detection
            is always disabled here.
    """
    assert Path(sc_path).exists(), f"sc.h5ad not found: {sc_path}"
    assert Path(st_path).exists(), f"st.h5ad not found: {st_path}"

    output_folder.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    logger.info("Using device: %s", device)

    logger.info("Load data")
    adata_sc = anndata.read_h5ad(Path(sc_path))  # C x G (raw counts)
    adata_st = anndata.read_h5ad(Path(st_path))  # S x G (raw counts)
    logger.info("scRNA: %d cells x %d genes", adata_sc.n_obs, adata_sc.n_vars)
    logger.info("Spatial: %d spots x %d genes", adata_st.n_obs, adata_st.n_vars)

    if cell_type_key not in adata_sc.obs.columns:
        raise KeyError(
            f"cell_type_key '{cell_type_key}' not found in obs columns "
            f"{list(adata_sc.obs.columns)}."
        )

    # --- Build the AnnData objects SPANN expects -----------------------------------
    # Reference: SPANN's anndata_preprocess derives the integer `labels` itself from
    # `cell_type`, so we only need `cell_type` (str) and `source`.
    adata_rna = adata_sc.copy()
    adata_rna.obs["cell_type"] = adata_rna.obs[cell_type_key].astype(str)
    adata_rna.obs["source"] = "scRNA"

    # Spatial: SPANN needs per-cell X/Y coordinates in obs.
    adata_spa = adata_st.copy()
    if "X" not in adata_spa.obs.columns or "Y" not in adata_spa.obs.columns:
        if "spatial" not in adata_spa.obsm:
            raise KeyError(
                "Spatial data needs coordinates: neither obs['X']/obs['Y'] nor "
                "obsm['spatial'] found."
            )
        coords = np.asarray(adata_spa.obsm["spatial"])
        adata_spa.obs["X"] = coords[:, 0]
        adata_spa.obs["Y"] = coords[:, 1]
    adata_spa.obs["source"] = "Spatial"

    # Remember the original spot order to realign SPANN's fresh output AnnData back to
    # it (SPANN builds adata_target with a default 0..N index, positionally aligned to
    # the spatial input).
    original_st_obs_names = adata_st.obs_names.astype(str)

    # --- SPANN pipeline ------------------------------------------------------------
    logger.info("Preprocess (anndata_preprocess)")
    adata_cm, adata_spa, adata_rna = anndata_preprocess(
        adata_spa, adata_rna, highly_variable=highly_variable, spatial_labels=False
    )

    logger.info("Build dataloaders")
    (
        source_sp_ds,
        target_sp_ds,
        source_cm_dl,
        target_cm_dl,
        test_source_cm_dl,
        test_target_cm_dl,
    ) = generate_dataloaders(adata_cm, adata_spa, adata_rna, batch_size=batch_size)

    logger.info("Construct SPANN model")
    enc, dec, x_dim, z_dim = generate_ae_params(
        adata_cm, adata_spa, adata_rna, feat_dim=feat_dim
    )
    cell_types = np.unique(adata_rna.obs["cell_type"])
    spann = SPANN_model(
        x_dim, z_dim, enc, dec, class_num=len(cell_types), device=torch_device
    )

    logger.info(
        "Train SPANN (%d cell types, novel detection disabled)", len(cell_types)
    )
    _, adata_target, _ = spann.train(
        source_cm_dl,
        target_cm_dl,
        source_sp_ds,
        target_sp_ds,
        adata_spa.obs[["X", "Y"]],
        test_source_cm_dl,
        test_target_cm_dl,
        np.array(adata_rna.obs["labels"]),
        cell_types,
        lr=lr,
        lambda_recon=lambda_recon,
        lambda_kl=lambda_kl,
        lambda_spa=lambda_spa,
        lambda_cd=lambda_cd,
        lambda_nb=lambda_nb,
        resolution=1.0,  # "no novel cells" -> every spot maps to a reference type
        novel_cell_test=False,  # don't let the dip test override resolution
        maxiter=maxiter,
        miditer1=miditer1,
        miditer2=miditer2,
        miditer3=miditer3,
    )

    # --- Assemble the (S x T) soft mapping -----------------------------------------
    # adata_target rows are positionally aligned to the (preprocessed) spatial input,
    # so index the hard predictions by that spot order, then one-hot over the reference
    # cell types and realign to the original ST spot order.
    pred_cls = np.asarray(adata_target.obs["pred_cls"])
    spot_names = adata_spa.obs_names.astype(str)

    n_unknown = int((pred_cls == "Unknown").sum())
    if n_unknown:
        logger.warning(
            "%d/%d spots labelled 'Unknown' despite resolution=1.0; "
            "they become all-zero rows (no reference type).",
            n_unknown,
            len(pred_cls),
        )

    onehot = pd.get_dummies(pd.Categorical(pred_cls, categories=cell_types)).astype(
        np.float32
    )
    onehot.index = spot_names
    onehot.columns = cell_types.astype(str)
    # Realign to the original ST spot order; any spot SPANN dropped comes back as zeros.
    onehot = onehot.reindex(index=original_st_obs_names, fill_value=0.0)

    ad_map = AnnData(
        X=onehot.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=onehot.index),
        var=pd.DataFrame(index=onehot.columns),
    )
    out_path = output_folder / "mapping_prob.h5ad"
    ad_map.write_h5ad(out_path)
    logger.info("Saved mapping to %s (shape %s)", out_path, ad_map.shape)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run SPANN alignment on a dataset")
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
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
        help="obs column on the reference to use as the cell-type annotation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default=None,
        help="Torch device (e.g. cuda / cuda:0 / cpu); auto-detected when omitted.",
    )
    args = parser.parse_args()

    spann_align_data(
        args.scdata,
        args.stdata,
        cell_type_key=args.cell_type_key,
        output_folder=args.output_folder,
        device=args.device,
    )
