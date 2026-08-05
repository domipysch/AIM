"""Reconstruction cosine similarity of the mapped spots vs. measured ST expression."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import anndata as ad
import pandas as pd

from adata_schema import (
    OBS_MAPPING_HARD,
    OBSM_LOGNORM_SHARED_GENES,
    OBSM_MAPPING_SOFT,
    UNS_LEIDEN_EXPR_SUMS_SHARED_GENES,
    UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM,
    UNS_LEIDEN_SIZES,
    UNS_SHARED_GENES,
)
from metrics import (
    CossimResult,
    assemble_state_centroids,
    compute_and_save_cossim,
    cossim_null_medians,
    predict_expression,
)

from .utils import to_dense

logger = logging.getLogger(__name__)


N_PERM_RECONSTRUCTION = 20  # label shuffles for the reconstruction null


def analyse_reconstruction(
    adata_sc: ad.AnnData,
    adata_st: ad.AnnData,
    leiden_to_state: np.ndarray,
    output_data_dir: Path,
    n_perm: int = N_PERM_RECONSTRUCTION,
    seed: int = 0,
) -> None:
    """Cosine similarity of predicted (``mapping @ state_centroids``) vs. measured
    spot expression, per gene and per spot, for the four soft/hard x raw/norm combos.

    Each *hard* combo also gets a label-shuffling null (``median_*_null``): the same
    medians with the spot->state labels permuted, averaged over ``n_perm`` shuffles.
    That is the chance level of the reconstruction — with only positive expression
    profiles in play it is far above zero, so it is the reference the K-sweep view
    compares against. The soft combos have no null (their assignment is a
    distribution, not a label) and come through as NaN.

    ``leiden_to_state`` is this K's subcluster->state cut (L,).
    Requires: adata_st.obsm[OBSM_MAPPING_SOFT], adata_st.obs[OBS_MAPPING_HARD],
        adata_st.obsm[OBSM_LOGNORM_SHARED_GENES], adata_sc.uns[UNS_SHARED_GENES],
        adata_sc.uns[UNS_LEIDEN_SIZES], adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES],
        adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM].
    Writes cossim_summary.csv and per-combo cossim JSONs under output_data_dir.
    """

    P = adata_st.obsm[OBSM_MAPPING_SOFT]
    spot_states_hard = adata_st.obs[OBS_MAPPING_HARD].to_numpy()
    k = P.shape[1]

    shared_genes = list(adata_sc.uns[UNS_SHARED_GENES])
    sizes = adata_sc.uns[UNS_LEIDEN_SIZES]

    # Measured ST expression on the shared genes: raw counts from a view, and the
    # normalize+log1p variant precomputed over the shared genes only.
    adata_st_shared = adata_st[:, shared_genes]
    adata_st_norm = ad.AnnData(
        X=to_dense(adata_st.obsm[OBSM_LOGNORM_SHARED_GENES]),
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=shared_genes),
    )

    centroids_raw = assemble_state_centroids(
        leiden_to_state, k, adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES], sizes
    )
    centroids_norm = assemble_state_centroids(
        leiden_to_state, k, adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM], sizes
    )

    # "hard" combos index each spot's winning-state centroid row directly;
    # "soft" combos are a genuine mapping @ centroids product.
    spot_names = adata_st.obs_names.tolist()
    combos = {
        "soft-raw": (P, centroids_raw, adata_st_shared, False),
        "hard-raw": (spot_states_hard, centroids_raw, adata_st_shared, True),
        "soft-norm": (P, centroids_norm, adata_st_norm, False),
        "hard-norm": (spot_states_hard, centroids_norm, adata_st_norm, True),
    }
    cossim_summary: dict[str, dict] = {}
    cossim_dir = output_data_dir / "cossim"
    cossim_results: dict[str, CossimResult] = {}
    rng = np.random.default_rng(seed)
    for label, (m, centroids, st_ref, is_hard) in combos.items():
        pred = centroids[m] if is_hard else predict_expression(m, centroids)
        pred_adata = ad.AnnData(
            X=pred.T.astype(np.float32),
            obs=pd.DataFrame(index=shared_genes),
            var=pd.DataFrame(index=spot_names),
        )
        result = compute_and_save_cossim(
            st_ref, pred_adata, cossim_dir, suffix=f"-{label}"
        )
        cossim_results[label] = result
        # ``st_ref`` is already restricted to the shared genes in centroid column
        # order, so the null reduces the very same matrices as the observed value.
        null = (
            cossim_null_medians(to_dense(st_ref), centroids, m, n_perm=n_perm, rng=rng)
            if label == "hard-raw"
            else {"median_spot": float("nan"), "median_gene": float("nan")}
        )
        cossim_summary[label] = {
            "median_gene": result.median_gene,
            "median_spot": result.median_spot,
            "median_gene_null": null["median_gene"],
            "median_spot_null": null["median_spot"],
        }

    pd.DataFrame(cossim_summary).T.to_csv(output_data_dir / "cossim_summary.csv")
