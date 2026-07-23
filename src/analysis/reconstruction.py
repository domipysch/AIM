"""Reconstruction cosine similarity for one K's post-mapping analysis.

Scores how well the mapping reconstructs the measured ST expression: predicted
spot expression is ``mapping @ state_centroids``, compared to the measured spots
by cosine similarity (per gene and per spot), for the four soft/hard x
raw/normalized combos.

Pure orchestration for the AIM analysis — it pulls the precomputed pieces off
``adata_sc``/``adata_st`` (per-subcluster expression sums / sizes, the soft
mapping P and its argmax) and delegates the actual math to ``metrics``
(``assemble_state_centroids`` / ``predict_expression`` / ``compute_and_save_cossim``).
"""

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
    predict_expression,
)
from plots import plot_cossim_boxplots

from .utils import to_dense

logger = logging.getLogger(__name__)


def analyse_reconstruction(
    adata_sc: ad.AnnData,
    adata_st: ad.AnnData,
    leiden_to_state: np.ndarray,
    output_data_dir: Path,
    output_plots_dir: Path,
) -> None:
    """Reconstruction cosine similarity of the mapped spots vs. their measured
    expression, for the four soft/hard x raw/normalized combos.

    Reads the precomputed soft mapping P and its argmax from ``adata_st.obsm``
    and the fixed per-subcluster expression sums / sizes from ``adata_sc.uns``;
    ``leiden_to_state`` is this K's subcluster->state tree cut (L,). Writes
    ``cossim_summary.csv`` + the per-combo cossim JSONs under ``output_data_dir``
    and the boxplot under ``output_plots_dir``.
    """

    # Precomputed upstream — read back off the adatas rather than recomputed.
    P = adata_st.obsm[OBSM_MAPPING_SOFT]  # (S x k) soft spot -> state mapping
    spot_states_hard = adata_st.obs[OBS_MAPPING_HARD].to_numpy()  # (S,)
    k = P.shape[1]

    shared_genes = list(adata_sc.uns[UNS_SHARED_GENES])
    sizes = adata_sc.uns[UNS_LEIDEN_SIZES]

    # Measured ST expression on the shared genes. Raw counts come from a view;
    # the normalized variant reuses the precomputed obsm[OBSM_LOGNORM_SHARED_GENES]
    # (normalize+log1p with the size factor over the shared genes only, written
    # once in aim.sweep.pre_processing). This is the ST counterpart of the exact
    # transform the sc _NORM centroids were built from (aim.aggregation reads the
    # same obsm), so the "norm" cossim compares like with like — not the
    # all-gene-normalized LAYER_LOGNORM sliced down, whose size factor differs.
    adata_st_shared = adata_st[:, shared_genes]  # view, not a copy
    adata_st_norm = ad.AnnData(
        X=to_dense(adata_st.obsm[OBSM_LOGNORM_SHARED_GENES]),
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=shared_genes),
    )

    # leiden_to_state is the only merge there is (the tree cut has no soft
    # form) — assemble each centroid set once and reuse it for both the
    # soft-P and hard-P cossim combos below, instead of recomputing it twice.
    centroids_raw = assemble_state_centroids(
        leiden_to_state, k, adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES], sizes
    )
    centroids_norm = assemble_state_centroids(
        leiden_to_state, k, adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM], sizes
    )

    # "hard" combos use spot_states_hard (winning-state index per spot) to
    # gather each spot's assigned-state centroid row directly — no one-hot
    # matrix needed, unlike "soft" which is a genuine mapping @ centroids.
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
    for label, (m, centroids, st_ref, is_hard) in combos.items():
        pred = (
            centroids[m] if is_hard else predict_expression(m, centroids)
        )  # S x G_shared
        pred_adata = ad.AnnData(
            X=pred.T.astype(np.float32),
            obs=pd.DataFrame(index=shared_genes),
            var=pd.DataFrame(index=spot_names),
        )
        result = compute_and_save_cossim(
            st_ref, pred_adata, cossim_dir, suffix=f"-{label}"
        )
        cossim_results[label] = result
        cossim_summary[label] = {
            "median_gene": result.median_gene,
            "median_spot": result.median_spot,
        }

    pd.DataFrame(cossim_summary).T.to_csv(output_data_dir / "cossim_summary.csv")
    plot_cossim_boxplots(cossim_results, output_plots_dir / "cossim_boxplots.png")
