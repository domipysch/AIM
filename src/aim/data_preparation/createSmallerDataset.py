import os
import anndata as ad
import numpy as np
from scipy.sparse import issparse

if __name__ == "__main__":
    """
    Creates a smaller subset dataset from a full scRNA-seq + ST dataset.
    Subsets cells, genes, and spots to configurable target counts, ensuring that
    selected cells and spots each express at least one shared gene.
    Reads from sc.h5ad + st.h5ad and writes sc.h5ad + st.h5ad.
    """

    dataset_from = "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_Datasets/03_MouseSSP"
    dataset_to = "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_Datasets/03_MouseSSP_Large"

    # Config for mini
    # target_nr_sc_cells = 15
    # target_nr_sc_genes = 30
    # target_nr_st_spots = 10
    # target_nr_st_genes = 20
    # target_nr_shared_genes = 5

    # Config for midi
    # target_nr_sc_cells = 100
    # target_nr_sc_genes = 300
    # target_nr_st_spots = 80
    # target_nr_st_genes = 240
    # target_nr_shared_genes = 15

    # Config for large
    target_nr_sc_cells = 1000
    target_nr_sc_genes = 5000
    target_nr_st_spots = 800
    target_nr_st_genes = 33
    target_nr_shared_genes = 33

    rng: np.random.Generator = (
        np.random.default_rng()
    )  # no fixed seed — sampling is non-deterministic across runs

    """ Load datasets """

    scData = ad.read_h5ad(os.path.join(dataset_from, "sc.h5ad"))  # C x G
    stData = ad.read_h5ad(os.path.join(dataset_from, "st.h5ad"))  # S x G

    """ Subset genes """

    # Split genes into shared and non-shared genes
    sharedGenes = scData.var_names.intersection(stData.var_names)
    scNonSharedGenes = scData.var_names.difference(sharedGenes)
    stNonSharedGenes = stData.var_names.difference(sharedGenes)

    # Create list of genes to keep (target_nr, half shared, half non-shared)
    scGenesToKeep = list(sharedGenes[:target_nr_shared_genes]) + list(
        scNonSharedGenes[: target_nr_sc_genes - target_nr_shared_genes]
    )
    if len(stNonSharedGenes) >= target_nr_shared_genes:
        stGenesToKeep = list(sharedGenes[:target_nr_shared_genes]) + list(
            stNonSharedGenes[: target_nr_st_genes - target_nr_shared_genes]
        )
    else:
        stGenesToKeep = list(sharedGenes[:target_nr_st_genes])

    """ Subset cells """

    # Select cells that express at least one shared gene
    sc_mat = scData[:, sharedGenes[:target_nr_shared_genes]].X
    if issparse(sc_mat):
        sc_mask = np.asarray(sc_mat.getnnz(axis=1) > 0).reshape(-1)
    else:
        sc_mask = np.any(sc_mat != 0, axis=1).reshape(-1)
    available_sc = scData.obs_names[sc_mask].tolist()
    n_sc = min(len(available_sc), target_nr_sc_cells)
    scCellsToKeep = rng.choice(available_sc, size=n_sc, replace=False).tolist()

    """ Subset spots """

    # Select spots that express at least one shared gene
    st_mat = stData[:, sharedGenes].X
    if issparse(st_mat):
        st_mask = np.asarray(st_mat.getnnz(axis=1) > 0).reshape(-1)
    else:
        st_mask = np.any(st_mat != 0, axis=1).reshape(-1)
    available_st = stData.obs_names[st_mask].tolist()
    n_st = min(len(available_st), target_nr_st_spots)
    stSpotsToKeep = rng.choice(available_st, size=n_st, replace=False).tolist()

    """ Create subsets """

    scData_mini = scData[scCellsToKeep, scGenesToKeep].copy()
    stData_mini = stData[stSpotsToKeep, stGenesToKeep].copy()

    """ Save to disk """

    os.makedirs(dataset_to, exist_ok=True)
    scData_mini.write_h5ad(os.path.join(dataset_to, "sc.h5ad"))
    stData_mini.write_h5ad(os.path.join(dataset_to, "st.h5ad"))

    print(f"Saved subset dataset to {dataset_to}")
    print(f"  scRNA: {scData_mini.n_obs} cells x {scData_mini.n_vars} genes")
    print(f"  ST:    {stData_mini.n_obs} spots x {stData_mini.n_vars} genes")
