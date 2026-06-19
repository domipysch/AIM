import os
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from anndata import AnnData
import argparse
from .utils.dataset_query import get_z_real_and_predicted_data_only_shared_genes
from .utils.distance_metrics import cosine_similarity
from .metrics_o4 import create_spatial_graph, NeighborhoodType
import json
import sys
import logging

logger = logging.getLogger(__name__)


NUM_PERMUTATIONS = 50


def add_own_metrics_to_edges(
    adata_z: AnnData, adata_predicted_z: AnnData, graph: nx.Graph
) -> nx.Graph:

    # helper to get dense 1D numpy array for a given spot id (obs_name)
    def _get_spot_vector(adata: AnnData, spot_id: str) -> np.ndarray:
        row_idx = int(np.where(np.asarray(adata.obs_names) == spot_id)[0][0])
        return np.asarray(adata.X[row_idx]).ravel().astype(float)

    # iterate over edges and annotate
    for u, v in graph.edges():
        su, sv = str(u), str(v)

        vec_u_z = _get_spot_vector(adata_z, su)
        vec_v_z = _get_spot_vector(adata_z, sv)
        vec_u_pred = _get_spot_vector(adata_predicted_z, su)
        vec_v_pred = _get_spot_vector(adata_predicted_z, sv)

        graph[u][v][f"cossim_z"] = float(cosine_similarity(vec_u_z, vec_v_z))
        graph[u][v][f"cossim_pred"] = float(cosine_similarity(vec_u_pred, vec_v_pred))

    return graph


def compute_permutation_test(edge_values: pd.DataFrame, output_folder: Path) -> None:

    # Compute actual T value as sum of abs(cossim_z - cossim_pred)
    T_actual = np.sum(np.abs(edge_values["cossim_z"] - edge_values["cossim_pred"]))

    # Compute permutation distribution
    T_perm_values = []
    for _ in range(NUM_PERMUTATIONS):
        # Shuffle predicted values
        shuffled_pred = (
            edge_values["cossim_pred"]
            .sample(frac=1, replace=False)
            .reset_index(drop=True)
        )
        T_perm = np.sum(np.abs(edge_values["cossim_z"] - shuffled_pred))
        T_perm_values.append(T_perm)

    print(f"Completed {NUM_PERMUTATIONS}/{NUM_PERMUTATIONS} permutations (per spot).")

    result = {"T_original": T_actual, "T_permuted": T_perm_values}

    result_file = output_folder / "permutation_test.json"
    with open(result_file, "w") as f:
        json.dump(result, f)


def add_p_value_to_json(json_path):
    """
    Read an existing permutation result JSON, compute empirical p-value from
    T_original and T_permuted if not present, and write it back.
    Empirical p-value (one-sided >=): (count(T_perm >= T_original) + 1) / (N + 1)
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} does not exist")

    with open(json_path, "r") as f:
        data = json.load(f)

    if "p_value" in data:
        print(f"p_value already present in {json_path}, skipping.")
        return

    if "T_original" not in data or "T_permuted" not in data:
        raise ValueError(f"{json_path} missing T_original or T_permuted")

    T_original = float(data["T_original"])
    T_permuted = np.asarray(data["T_permuted"], dtype=float).ravel()
    if T_permuted.size == 0:
        raise ValueError(f"{json_path} contains empty T_permuted")

    less_equal_count = int(np.sum(T_permuted <= T_original))
    p_value = (less_equal_count + 1) / (T_permuted.size + 1)

    # ensure JSON serializable simple float
    data["p_value"] = float(p_value)

    # overwrite file
    with open(json_path, "w") as f:
        json.dump(data, f)

    print(f"Added p_value={p_value:.6f} to {json_path}")


def main(sc_path: Path, st_path: Path, result_gep: AnnData, metrics_folder_name: Path):
    logger.info("Run permutation test for objective o4")

    output_folder = metrics_folder_name / "o4" / "knn"
    os.makedirs(output_folder, exist_ok=True)

    # S x shared_genes
    adata_z, adata_predicted_z = get_z_real_and_predicted_data_only_shared_genes(
        sc_path, st_path, result_gep
    )
    # Assert that both DataFrames have the same shape of genes and spots
    assert (
        adata_z.shape == adata_predicted_z.shape
    ), "DataFrames haben unterschiedliche Formen."
    assert adata_z.n_obs == result_gep.n_vars

    # KNN
    graph = create_spatial_graph(st_path, neighborhood_type=NeighborhoodType.KNN, k=4)
    graph = add_own_metrics_to_edges(
        adata_z,
        adata_predicted_z,
        graph,
    )

    # Collect edge metrics into a DataFrame
    edge_data = []
    for u, v, data in graph.edges(data=True):
        edge_entry = {
            "spot_u": u,
            "spot_v": v,
            "cossim_z": data.get("cossim_z", np.nan),
            "cossim_pred": data.get("cossim_pred", np.nan),
        }
        edge_data.append(edge_entry)
    edge_df = pd.DataFrame(edge_data)
    compute_permutation_test(edge_df, output_folder)

    add_p_value_to_json(output_folder / "permutation_test.json")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run permutation test on custom metric 4"
    )
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument("-r", "--result", type=Path, help="Path to result file")
    parser.add_argument(
        "-m", "--metrics", type=Path, help="Path to output metric folder"
    )
    args = parser.parse_args()

    main(args.scdata, args.stdata, args.result, args.metrics)
