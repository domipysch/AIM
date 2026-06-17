from pathlib import Path
import anndata
import numpy as np
import logging
import json
from anndata import AnnData
from .utils.dataset_query import get_z_real_and_predicted_data_only_shared_genes
from .utils.distance_metrics import cosine_similarity

logger = logging.getLogger(__name__)


def compute_metrics_per_gene(
    adata_z, adata_predicted_z, save_cossim_json: Path | None = None
) -> None:
    """
    Compute cosine similarity (or other distance metrics) per gene and write results directly into
    adata_predicted_z.var['cossim'].

    Optional: if save_cossim_json is provided (Path), the per-gene cossim values
    are also written to a JSON file (gene -> float).
    """

    # initialize column with NaN if not already present
    adata_predicted_z.var["cossim"] = np.nan
    # adata_predicted_z.var['sqrt_cossim'] = np.nan
    # adata_predicted_z.var['eucl'] = np.nan
    # adata_predicted_z.var['rmse'] = np.nan
    # adata_predicted_z.var['mae'] = np.nan
    # adata_predicted_z.var['canberra'] = np.nan
    # adata_predicted_z.var['pearson'] = np.nan
    # adata_predicted_z.var['bray_curtis'] = np.nan
    # adata_predicted_z.var['aitchison'] = np.nan
    # adata_predicted_z.var['kl'] = np.nan
    # adata_predicted_z.var['js'] = np.nan
    # adata_predicted_z.var['hellinger'] = np.nan
    # adata_predicted_z.var['bhat'] = np.nan
    # adata_predicted_z.var['tv'] = np.nan

    # Collect cossim values for optional export
    cossim_dict = {}

    counter = 0
    for gene in adata_predicted_z.var_names:

        # Retrieve vectors (AnnData slicing may return 2D arrays)
        vec_z = adata_z[:, gene].X.toarray().ravel()
        vec_pred = adata_predicted_z[:, gene].X.toarray().ravel()

        # Compute and store values in local variables before assignment (useful for export)
        val_cossim = float(cosine_similarity(vec_z, vec_pred))
        # val_sqrt_cossim = float(sqrt_cosine_similarity(vec_z, vec_pred))
        # val_eucl = float(euclidean_l2(vec_z, vec_pred))
        # val_rmse = float(rmse(vec_z, vec_pred))
        # val_mae = float(mae_l1(vec_z, vec_pred))
        # val_canberra = float(canberra(vec_z, vec_pred))
        # val_pearson = float(pearson_distance(vec_z, vec_pred))
        # val_bray = float(bray_curtis_distance(vec_z, vec_pred))
        # val_aitchison = float(aitchison_distance(vec_z, vec_pred))
        # val_kl = float(kl_divergence(vec_z, vec_pred))
        # val_js = float(jensen_shannon_distance(vec_z, vec_pred))
        # val_hell = float(hellinger_distance(vec_z, vec_pred))
        # val_bhat = float(bhattacharyya_distance(vec_z, vec_pred))
        # val_tv = float(total_variation(vec_z, vec_pred))

        # Store directly in adata_predicted_z.var
        adata_predicted_z.var.at[gene, "cossim"] = val_cossim
        # adata_predicted_z.var.at[gene, 'sqrt_cossim'] = val_sqrt_cossim
        # adata_predicted_z.var.at[gene, 'eucl'] = val_eucl
        # adata_predicted_z.var.at[gene, 'rmse'] = val_rmse
        # adata_predicted_z.var.at[gene, 'mae'] = val_mae
        # adata_predicted_z.var.at[gene, 'canberra'] = val_canberra
        # adata_predicted_z.var.at[gene, 'pearson'] = val_pearson
        # adata_predicted_z.var.at[gene, 'bray_curtis'] = val_bray
        # adata_predicted_z.var.at[gene, 'aitchison'] = val_aitchison
        # adata_predicted_z.var.at[gene, 'kl'] = val_kl
        # adata_predicted_z.var.at[gene, 'js'] = val_js
        # adata_predicted_z.var.at[gene, 'hellinger'] = val_hell
        # adata_predicted_z.var.at[gene, 'bhat'] = val_bhat
        # adata_predicted_z.var.at[gene, 'tv'] = val_tv

        # Collect for export
        cossim_dict[gene] = val_cossim

        counter += 1
        if counter % 1000 == 0:
            logging.info(f"Processed {counter}/{adata_predicted_z.n_vars} genes.")

    # Optional: save cossim per gene as JSON
    if save_cossim_json is not None:
        save_cossim_json.parent.mkdir(parents=True, exist_ok=True)
        values = list(cossim_dict.values())
        output = {
            "median": float(np.median(values)) if values else None,
            "values": cossim_dict,
        }
        with save_cossim_json.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)


def compute_metrics_per_spot(
    adata_z, adata_predicted_z, save_cossim_json: Path | None = None
) -> None:
    """
    Compute cosine similarity (or other distance metrics) per spot and write results directly into
    adata_predicted_z.obs['cossim'].
    """

    # initialize obs columns with NaN if not already present
    adata_predicted_z.obs["cossim"] = np.nan
    # adata_predicted_z.obs['sqrt_cossim'] = np.nan
    # adata_predicted_z.obs['eucl'] = np.nan
    # adata_predicted_z.obs['rmse'] = np.nan
    # adata_predicted_z.obs['mae'] = np.nan
    # adata_predicted_z.obs['canberra'] = np.nan
    # adata_predicted_z.obs['pearson'] = np.nan
    # adata_predicted_z.obs['bray_curtis'] = np.nan
    # adata_predicted_z.obs['aitchison'] = np.nan
    # adata_predicted_z.obs['kl'] = np.nan
    # adata_predicted_z.obs['js'] = np.nan
    # adata_predicted_z.obs['hellinger'] = np.nan
    # adata_predicted_z.obs['bhat'] = np.nan
    # adata_predicted_z.obs['tv'] = np.nan

    # Collect cossim values for optional export
    cossim_dict = {}

    counter = 0
    for spot in adata_predicted_z.obs_names:
        # Retrieve vectors (AnnData slicing may return 2D arrays)
        vec_z = adata_z[spot, :].X.toarray().ravel()
        vec_pred = adata_predicted_z[spot, :].X.toarray().ravel()

        # Store metrics directly in adata_predicted_z.obs
        val_cossim = cosine_similarity(vec_z, vec_pred)
        adata_predicted_z.obs.at[spot, "cossim"] = val_cossim
        # adata_predicted_z.obs.at[spot, 'sqrt_cossim'] = sqrt_cosine_similarity(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'eucl'] = euclidean_l2(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'rmse'] = rmse(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'mae'] = mae_l1(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'canberra'] = canberra(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'pearson'] = pearson_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'bray_curtis'] = bray_curtis_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'aitchison'] = aitchison_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'kl'] = kl_divergence(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'js'] = jensen_shannon_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'hellinger'] = hellinger_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'bhat'] = bhattacharyya_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'tv'] = total_variation(vec_z, vec_pred)
        cossim_dict[spot] = val_cossim

        counter += 1
        if counter % 1000 == 0:
            logging.info(f"Processed {counter}/{adata_predicted_z.n_obs} spots.")

    # Optional: save cossim per spot as JSON
    if save_cossim_json is not None:
        save_cossim_json.parent.mkdir(parents=True, exist_ok=True)
        values = list(cossim_dict.values())
        output = {
            "median": float(np.median(values)) if values else None,
            "values": cossim_dict,
        }
        with save_cossim_json.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)


def main(
    sc_path: Path,
    st_path: Path,
    result_gep: AnnData,
    metrics_output_folder: Path,
    name_suffix: str = "",
):
    """
    Compute metrics for objective 3 and save results as JSON files / diagrams.

    - Compute cosine similarity (and other distance metrics) per gene and save to json;
    - Compute cosine similarity (and other distance metrics) per spot and save to json;

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        result_gep: G x S predicted gene expression AnnData
        metrics_output_folder: Folder where JSON files are written directly.
        name_suffix: Appended to output filenames before .json (e.g. "-det").

    Returns: None
    """
    logger.info("Compute metrics for o2")

    metrics_output_folder.mkdir(parents=True, exist_ok=True)

    metrics_cossim_per_gene_json = (
        metrics_output_folder / f"cossim-per-gene{name_suffix}.json"
    )
    metrics_cossim_per_spot_json = (
        metrics_output_folder / f"cossim-per-spot{name_suffix}.json"
    )

    # Load data (input ST and predicted)
    adata_z, adata_predicted_z = get_z_real_and_predicted_data_only_shared_genes(
        sc_path, st_path, result_gep
    )

    # Assert that both DataFrames have the same shape of genes and spots
    assert (
        adata_z.shape == adata_predicted_z.shape
    ), "DataFrames haben unterschiedliche Formen."

    # Add spatial locations to AnnData objects
    adata_st_full = anndata.read_h5ad(st_path)
    spatial = adata_st_full.obsm["spatial"]
    spot_index = {name: i for i, name in enumerate(adata_st_full.obs_names)}
    coords = np.array([spatial[spot_index[s]] for s in adata_z.obs_names])
    adata_z.obsm["coords"] = coords
    adata_predicted_z.obsm["coords"] = coords

    # Compute and store cossim per gene in adata_predicted_z.var
    compute_metrics_per_gene(
        adata_z, adata_predicted_z, save_cossim_json=metrics_cossim_per_gene_json
    )

    # Compute and store cossim per spot in adata_predicted_z.var
    compute_metrics_per_spot(
        adata_z, adata_predicted_z, save_cossim_json=metrics_cossim_per_spot_json
    )
