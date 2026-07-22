"""
Per-K disk outputs for the agglomerative sweep.

Each K folder is written in the exact layout the post-mapping analysis expects,
so ``analysis.run_from_output`` consumes it unchanged:

    spot_to_state_mapping.h5ad — P, obs = spots, var = computed states (S x K)
    spot_to_state_mapping.csv  — P as a CSV (thresholded, rounded) for eyeballing
    leiden_to_state.csv        — subcluster -> state map (labels_k); the hard
                                 tree cut is used directly as this label array,
                                 never materialized as a one-hot matrix

``leiden_overclustering.h5ad`` (per-cell Leiden label) does not vary with K —
it is written once per run, in the run's root folder, by
``write_leiden_overclustering``.
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

from adata_schema import OBS_LEIDEN_ALL_GENES, UNS_LEIDEN_NUMBER_STATES_ALL_GENES


def write_leiden_overclustering_all_genes(
    output_folder: Path, adata_sc: anndata.AnnData
) -> None:
    """
    Write the per-cell Leiden cluster label (obs["leiden_cluster"]) once for the
    whole run, in ``output_folder`` (the run root, not a per-K folder) — the
    Leiden over-clustering is computed once and reused by every K.
    """
    leiden_names = [
        f"leiden_{i}" for i in range(adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_ALL_GENES])
    ]
    leiden_labels = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    cell_cluster_names = [leiden_names[i] for i in leiden_labels]
    anndata.AnnData(
        X=np.zeros((len(cell_cluster_names), 0), dtype=np.float32),
        obs=pd.DataFrame(
            {"leiden_cluster": cell_cluster_names}, index=adata_sc.obs_names
        ),
    ).write_h5ad(output_folder / "leiden_overclustering.h5ad")


def write_run_outputs(
    run_dir: Path,
    spot_to_state: torch.Tensor,
    labels_k: np.ndarray,
    n_leiden: int,
    k: int,
    adata_st: anndata.AnnData,
) -> None:
    """Write one K's core outputs."""

    # Save used leiden cluster aggregation for this run
    pd.DataFrame({"leiden_cluster": np.arange(n_leiden), "state": labels_k}).to_csv(
        run_dir / "leiden_to_state.csv", index=False
    )

    # Saved spot to state mapping as h5ad and csv (at csv: rounded)
    state_names = [f"state_{i}" for i in range(k)]

    anndata.AnnData(
        X=spot_to_state.numpy(),
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=state_names),
    ).write_h5ad(run_dir / "spot_to_state_mapping.h5ad")

    p_csv = spot_to_state.numpy()
    p_csv[p_csv < 0.001] = 0.0
    pd.DataFrame(
        p_csv.round(4),
        index=adata_st.obs_names.tolist(),
        columns=state_names,
    ).to_csv(run_dir / "spot_to_state_mapping.csv")
