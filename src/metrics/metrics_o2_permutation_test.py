from pathlib import Path
import os
import numpy as np
import json
import multiprocessing as mp
import logging
from anndata import AnnData
from .utils.dataset_query import get_z_real_and_predicted_data_only_shared_genes

logger = logging.getLogger(__name__)


NUM_PERMUTATIONS = 50

# ------------- Per gene

# New helper functions for parallel computation
_Z_MATRIX = None
_PRED_MATRIX = None
_Z_NORMS = None
_N_OBS = None


def _init_worker(z_mat, pred_mat, z_norms):
    # Set global variables in each worker (serialized once per worker)
    global _Z_MATRIX, _PRED_MATRIX, _Z_NORMS, _N_OBS
    _Z_MATRIX = z_mat
    _PRED_MATRIX = pred_mat
    _Z_NORMS = z_norms
    _N_OBS = z_mat.shape[0]


def _compute_T_for_permutation(permuted_indices):
    # permuted_indices: 1D ndarray of length n_obs
    global _Z_MATRIX, _PRED_MATRIX, _Z_NORMS
    # Create permuted predicted matrix (same shape)
    pred_perm = _PRED_MATRIX[permuted_indices, :]
    # Vectorized computation: dot per gene and norms
    # dot per gene:
    dot_per_gene = np.sum(_Z_MATRIX * pred_perm, axis=0)  # shape: (n_genes,)
    pred_norms = np.linalg.norm(pred_perm, axis=0)
    # numerical stability
    denom = _Z_NORMS * pred_norms + 1e-12
    cosines = dot_per_gene / denom
    T = float(np.sum(cosines))
    return T


def permutation_test_per_gene(adata_z, adata_predicted_z, output_folder: Path) -> None:

    # Convert to numpy arrays (handles sparse matrices)
    z_matrix = (
        adata_z.X.toarray() if hasattr(adata_z.X, "toarray") else np.asarray(adata_z.X)
    )
    pred_matrix = (
        adata_predicted_z.X.toarray()
        if hasattr(adata_predicted_z.X, "toarray")
        else np.asarray(adata_predicted_z.X)
    )

    z_norms = np.linalg.norm(z_matrix, axis=0)
    # original T:
    dot_orig = np.sum(z_matrix * pred_matrix, axis=0)
    denom_orig = z_norms * np.linalg.norm(pred_matrix, axis=0) + 1e-12
    T_original = float(np.sum(dot_orig / denom_orig))
    logging.info(f"Original T value (per gene): {T_original}")

    # Perform permutation test
    n_obs = z_matrix.shape[0]

    # Generate permutations (list of index-arrays over spots)
    permutations = [np.random.permutation(n_obs) for _ in range(NUM_PERMUTATIONS)]

    # Start pool and initialize workers (each matrix is sent once per worker)
    cpu_count = max(1, os.cpu_count() or 1)
    with mp.Pool(
        processes=cpu_count,
        initializer=_init_worker,
        initargs=(z_matrix, pred_matrix, z_norms),
    ) as pool:
        # map the permutations to the workers
        T_permuted = pool.map(_compute_T_for_permutation, permutations)

    logging.info(
        f"Completed {NUM_PERMUTATIONS}/{NUM_PERMUTATIONS} permutations (per gene)."
    )

    # compute empirical p-value (one-sided: prob T_perm >= T_original) ---
    greater_equal_count = sum(1 for t in T_permuted if t >= T_original)
    p_value = (greater_equal_count + 1) / (len(T_permuted) + 1)  # +1 for continuity
    logging.info(f"Empirical p-value (per gene, one-sided >=): {p_value:.3f}")

    # Write json with original and permuted T values
    result = {
        "p_value": p_value,
        "T_original": T_original,
        "T_permuted": T_permuted,
    }

    # Save result as json
    result_file = output_folder / "permutation_test_per_gene.json"
    with open(result_file, "w") as f:
        json.dump(result, f)


# ------------- Per spot

# New global variables and worker functions for per-spot permutation
_Z_MATRIX_SPOT = None
_PRED_MATRIX_SPOT = None
_Z_NORMS_SPOT = None
_N_GENES = None


def _init_worker_spot(z_mat, pred_mat, z_norms):
    global _Z_MATRIX_SPOT, _PRED_MATRIX_SPOT, _Z_NORMS_SPOT, _N_GENES
    _Z_MATRIX_SPOT = z_mat
    _PRED_MATRIX_SPOT = pred_mat
    _Z_NORMS_SPOT = z_norms
    _N_GENES = z_mat.shape[1]


def _compute_T_for_permutation_spot(permuted_gene_indices):
    global _Z_MATRIX_SPOT, _PRED_MATRIX_SPOT, _Z_NORMS_SPOT
    # Permute gene/column order in predicted matrix
    pred_perm = _PRED_MATRIX_SPOT[:, permuted_gene_indices]
    # dot per spot (rows)
    dot_per_spot = np.sum(_Z_MATRIX_SPOT * pred_perm, axis=1)  # shape: (n_spots,)
    pred_norms = np.linalg.norm(pred_perm, axis=1)
    denom = _Z_NORMS_SPOT * pred_norms + 1e-12
    cosines = dot_per_spot / denom
    T = float(np.sum(cosines))
    return T


def permutation_test_per_spot(adata_z, adata_predicted_z, output_folder: Path) -> None:

    # Convert to numpy arrays (handles sparse matrices)
    z_matrix = (
        adata_z.X.toarray() if hasattr(adata_z.X, "toarray") else np.asarray(adata_z.X)
    )
    pred_matrix = (
        adata_predicted_z.X.toarray()
        if hasattr(adata_predicted_z.X, "toarray")
        else np.asarray(adata_predicted_z.X)
    )

    # norms per spot (rows)
    z_norms = np.linalg.norm(z_matrix, axis=1)
    # original T per spot:
    dot_orig = np.sum(z_matrix * pred_matrix, axis=1)
    denom_orig = z_norms * np.linalg.norm(pred_matrix, axis=1) + 1e-12
    T_original = float(np.sum(dot_orig / denom_orig))
    logging.info(f"Original T value (per spot): {T_original}")

    # Perform permutation test (permute gene indices)
    n_genes = z_matrix.shape[1]

    permutations = [np.random.permutation(n_genes) for _ in range(NUM_PERMUTATIONS)]

    cpu_count = max(1, os.cpu_count() or 1)
    with mp.Pool(
        processes=cpu_count,
        initializer=_init_worker_spot,
        initargs=(z_matrix, pred_matrix, z_norms),
    ) as pool:
        T_permuted = pool.map(_compute_T_for_permutation_spot, permutations)

    logging.info(
        f"Completed {NUM_PERMUTATIONS}/{NUM_PERMUTATIONS} permutations (per spot)."
    )

    # compute empirical p-value (one-sided: prob T_perm >= T_original) ---
    greater_equal_count = sum(1 for t in T_permuted if t >= T_original)
    p_value = (greater_equal_count + 1) / (len(T_permuted) + 1)  # +1 for continuity
    logging.info(f"Empirical p-value (per gene, one-sided >=): {p_value:.3f}")

    result = {"p_value": p_value, "T_original": T_original, "T_permuted": T_permuted}

    result_file = output_folder / "permutation_test_per_spot.json"
    with open(result_file, "w") as f:
        json.dump(result, f)


# ------------- Main and helper functions


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
        logging.info(f"p_value already present in {json_path}, skipping.")
        return

    if "T_original" not in data or "T_permuted" not in data:
        raise ValueError(f"{json_path} missing T_original or T_permuted")

    T_original = float(data["T_original"])
    T_permuted = np.asarray(data["T_permuted"], dtype=float).ravel()
    if T_permuted.size == 0:
        raise ValueError(f"{json_path} contains empty T_permuted")

    greater_equal_count = int(np.sum(T_permuted >= T_original))
    p_value = (greater_equal_count + 1) / (T_permuted.size + 1)

    # ensure JSON serializable simple float
    data["p_value"] = float(p_value)

    # overwrite file
    with open(json_path, "w") as f:
        json.dump(data, f)

    logging.info(f"Added p_value={p_value:.6f} to {json_path}")


def main(sc_path: Path, st_path: Path, result_gep: AnnData, metrics_folder_name: Path):
    logger.info("Run permutation test for objective o2")

    result_folder_permutation = metrics_folder_name / "o2" / "permutation_test"
    os.makedirs(result_folder_permutation, exist_ok=True)

    # Load data (S x shared G)
    adata_z, adata_predicted_z = get_z_real_and_predicted_data_only_shared_genes(
        sc_path, st_path, result_gep
    )

    # Assert that both DataFrames have the same shape of genes and spots
    assert (
        adata_z.shape == adata_predicted_z.shape
    ), "DataFrames haben unterschiedliche Formen."
    assert adata_z.n_obs == result_gep.n_vars

    permutation_test_per_gene(adata_z, adata_predicted_z, result_folder_permutation)
    # permutation_test_per_spot(adata_z, adata_predicted_z, result_folder_permutation)

    add_p_value_to_json(result_folder_permutation / "permutation_test_per_gene.json")
    # add_p_value_to_json(result_folder_permutation / "permutation_test_per_spot.json")
