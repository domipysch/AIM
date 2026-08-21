# Creates a derived dataset containing only the top N highly variable genes (HVGs)
# from the scRNA-seq data. ST data is copied unchanged. Shared genes that are
# not among the top HVGs are appended to the selection to preserve cross-modal overlap.
# Reads from sc.h5ad + st.h5ad and writes sc.h5ad + st.h5ad.

import os
import shutil
import anndata as ad
import scanpy as sc

if __name__ == "__main__":

    dataset_from = "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_Datasets/03_MouseSSP"
    dataset_to = (
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_Datasets/03_MouseSSP_HVG_2000"
    )
    N = 2000

    """ Load datasets """

    scData = ad.read_h5ad(os.path.join(dataset_from, "sc.h5ad"))  # C x G
    print("scRNA Data loaded")
    stData = ad.read_h5ad(os.path.join(dataset_from, "st.h5ad"))  # S x G
    print("ST Data loaded")

    """ Select top N HVGs, keeping all shared genes """

    # Shared genes between scRNA and ST
    sharedGenes = scData.var_names.intersection(stData.var_names)

    # Compute HVGs on a normalized log-transformed copy
    scDataLog = scData.copy()
    sc.pp.normalize_total(scDataLog)
    sc.pp.log1p(scDataLog)
    sc.pp.highly_variable_genes(scDataLog, n_top_genes=N, inplace=True)

    assert "highly_variable" in scDataLog.var.columns
    hvgs = list(scDataLog.var_names[scDataLog.var["highly_variable"].values])

    if len(hvgs) == 0:
        raise RuntimeError("No HVGs selected — check input data and N value")

    # Append shared genes that were not selected as HVGs
    for sharedGene in sharedGenes:
        if sharedGene not in hvgs:
            print(
                f"Warning: Shared gene {sharedGene} not selected as HVG - added manually"
            )
            hvgs.append(sharedGene)
    print(
        f"Appended {len(hvgs) - N} shared genes to HVG list. Total genes selected:",
        len(hvgs),
    )

    """ Create and save HVG-filtered scRNA dataset """

    scData_HVG = scData[:, hvgs].copy()

    os.makedirs(dataset_to, exist_ok=True)
    scData_HVG.write_h5ad(os.path.join(dataset_to, "sc.h5ad"))
    print(f"Saved sc.h5ad: {scData_HVG.n_obs} cells x {scData_HVG.n_vars} genes")

    """ Copy ST dataset unchanged """

    shutil.copy(
        os.path.join(dataset_from, "st.h5ad"),
        os.path.join(dataset_to, "st.h5ad"),
    )
    print("Copied st.h5ad unchanged")
