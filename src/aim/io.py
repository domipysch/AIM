"""Disk outputs for the sweep: the per-cell Leiden over-clustering (once per run) and,
per K, the spot->state mapping P (h5ad + CSV) and the subcluster->state label map."""

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
    Write leiden_overclustering.h5ad (obs["leiden_cluster"] per cell) to the run root.

    Requires: adata_sc.uns[UNS_LEIDEN_NUMBER_STATES_ALL_GENES], adata_sc.obs[OBS_LEIDEN_ALL_GENES].
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
    """Write one K's outputs: leiden_to_state.csv and spot_to_state_mapping.{h5ad,csv} (P is S x K)."""

    pd.DataFrame({"leiden_cluster": np.arange(n_leiden), "state": labels_k}).to_csv(
        run_dir / "leiden_to_state.csv", index=False
    )

    state_names = [f"state_{i}" for i in range(k)]

    anndata.AnnData(
        X=spot_to_state.numpy(),
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=state_names),
    ).write_h5ad(run_dir / "spot_to_state_mapping.h5ad")

    # The CSV copy is thresholded and rounded for readability.
    p_csv = spot_to_state.numpy()
    p_csv[p_csv < 0.001] = 0.0
    pd.DataFrame(
        p_csv.round(4),
        index=adata_st.obs_names.tolist(),
        columns=state_names,
    ).to_csv(run_dir / "spot_to_state_mapping.csv")
