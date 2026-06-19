"""
Dataset summary: per-gene statistics table and dataset-level overview.

Saves two files to output_dir:
  - summary.json   : dataset-level numbers
  - gene_table.csv : per-gene stats for all shared genes
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from .metric import to_dense

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering quality helpers
# ---------------------------------------------------------------------------


def _silhouette(X: np.ndarray, labels: np.ndarray, max_samples: int = 5000) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X, labels = X[idx], labels[idx]
    return round(float(silhouette_score(X, labels)), 4)


def _dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Centroid-based Dunn index approximation:
      inter-cluster = min distance between any two cluster centroids
      intra-cluster = max mean distance from points to their centroid
    Higher = more separated, more compact clusters.
    """
    unique = sorted(np.unique(labels))
    if len(unique) < 2:
        return float("nan")

    centroids = np.stack([X[labels == k].mean(axis=0) for k in unique])

    inter = euclidean_distances(centroids)
    np.fill_diagonal(inter, np.inf)
    min_inter = float(inter.min())

    max_intra = max(
        float(np.linalg.norm(X[labels == k] - centroids[i], axis=1).mean())
        for i, k in enumerate(unique)
    )

    return round(min_inter / max_intra, 4) if max_intra > 0 else float("nan")


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_dataset_summary(
    sc_adata: AnnData,
    st_adata: AnnData,
    X_sc: np.ndarray,
    X_st: np.ndarray,
    labels_sc: np.ndarray,
    labels_st: np.ndarray,
    shared: list[str],
    n_clusters_sc: int,
    n_clusters_st: int,
) -> dict:
    """Dataset-level stats. X_sc / X_st are already filtered to shared genes."""
    sc_libsize = to_dense(sc_adata.X).sum(axis=1)
    st_libsize = to_dense(st_adata.X).sum(axis=1)

    logger.info("Computing silhouette scores ...")
    sil_sc = _silhouette(X_sc, labels_sc)
    sil_st = _silhouette(X_st, labels_st)

    logger.info("Computing Dunn indices ...")
    dunn_sc = _dunn_index(X_sc, labels_sc)
    dunn_st = _dunn_index(X_st, labels_st)

    return {
        # cell / spot counts
        "n_cells_sc": int(sc_adata.n_obs),
        "n_spots_st": int(st_adata.n_obs),
        # gene counts
        "n_genes_sc": int(sc_adata.n_vars),
        "n_genes_st": int(st_adata.n_vars),
        "n_shared_genes": len(shared),
        "pct_shared_of_sc": round(100.0 * len(shared) / sc_adata.n_vars, 2),
        "pct_shared_of_st": round(100.0 * len(shared) / st_adata.n_vars, 2),
        # library size (all genes)
        "sc_libsize_mean": round(float(sc_libsize.mean()), 2),
        "sc_libsize_median": round(float(np.median(sc_libsize)), 2),
        "sc_libsize_std": round(float(sc_libsize.std()), 2),
        "st_libsize_mean": round(float(st_libsize.mean()), 2),
        "st_libsize_median": round(float(np.median(st_libsize)), 2),
        "st_libsize_std": round(float(st_libsize.std()), 2),
        # sparsity on shared genes
        "sc_sparsity": round(float((X_sc == 0).mean()), 4),
        "st_sparsity": round(float((X_st == 0).mean()), 4),
        # clustering
        "n_clusters_sc": n_clusters_sc,
        "n_clusters_st": n_clusters_st,
        # clustering quality (on shared genes)
        "silhouette_sc": sil_sc,
        "silhouette_st": sil_st,
        "dunn_sc": dunn_sc,
        "dunn_st": dunn_st,
    }


def compute_gene_table(
    X_sc: np.ndarray, X_st: np.ndarray, shared: list[str]
) -> pd.DataFrame:
    """
    Per-gene statistics DataFrame for all shared genes.

    Columns
    -------
    gene, mean_sc, mean_st, var_sc, var_st,
    dropout_sc, dropout_st,          # fraction of zeros
    cv_sc, cv_st,                    # std / mean (nan when mean == 0)
    var_rank_sc, var_rank_st,        # 1 = most variable
    var_rank_diff                    # |var_rank_sc - var_rank_st|
    """
    mean_sc = X_sc.mean(axis=0)
    mean_st = X_st.mean(axis=0)
    var_sc = X_sc.var(axis=0)
    var_st = X_st.var(axis=0)
    dropout_sc = (X_sc == 0).mean(axis=0)
    dropout_st = (X_st == 0).mean(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        cv_sc = np.where(mean_sc > 0, np.sqrt(var_sc) / mean_sc, np.nan)
        cv_st = np.where(mean_st > 0, np.sqrt(var_st) / mean_st, np.nan)

    n = len(shared)
    # rank 1 = most variable (highest variance)
    var_rank_sc = n + 1 - pd.Series(var_sc).rank(method="average").values
    var_rank_st = n + 1 - pd.Series(var_st).rank(method="average").values
    var_rank_diff = np.abs(var_rank_sc - var_rank_st)

    return pd.DataFrame(
        {
            "gene": shared,
            "mean_sc": mean_sc.round(4),
            "mean_st": mean_st.round(4),
            "var_sc": var_sc.round(4),
            "var_st": var_st.round(4),
            "dropout_sc": dropout_sc.round(4),
            "dropout_st": dropout_st.round(4),
            "cv_sc": np.round(cv_sc, 4),
            "cv_st": np.round(cv_st, 4),
            "var_rank_sc": var_rank_sc.astype(int),
            "var_rank_st": var_rank_st.astype(int),
            "var_rank_diff": var_rank_diff.astype(int),
        }
    ).sort_values("var_rank_diff", ascending=False)


# ---------------------------------------------------------------------------
# Cluster top-gene tables
# ---------------------------------------------------------------------------


def compute_cluster_top_genes(
    X: np.ndarray,
    labels: np.ndarray,
    gene_names: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """
    For each cluster: top-k genes by z-score of the cluster centroid.

    Z-score is computed across genes within the centroid vector (same normalization
    as used in the centroid cosine similarity metric and the match plots).

    Returns a long-format DataFrame with columns: cluster, rank, gene, zscore.
    """
    gene_arr = np.array(gene_names)
    rows = []
    for k in sorted(np.unique(labels)):
        centroid = X[labels == k].mean(axis=0)
        s = centroid.std()
        z = (centroid - centroid.mean()) / s if s > 0 else centroid - centroid.mean()
        top_idx = np.argsort(z)[::-1][:top_k]
        for rank, idx in enumerate(top_idx, 1):
            rows.append(
                {
                    "cluster": int(k),
                    "rank": rank,
                    "gene": gene_arr[idx],
                    "zscore": round(float(z[idx]), 4),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_summary(
    dataset_summary: dict,
    gene_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(dataset_summary, f, indent=2)
    logger.info(f"Saved dataset summary → {json_path}")

    csv_path = output_dir / "gene_table.csv"
    gene_table.to_csv(csv_path, index=False)
    logger.info(f"Saved gene table ({len(gene_table)} genes) → {csv_path}")


def save_cluster_tables(
    cluster_df_sc: pd.DataFrame,
    cluster_df_st: pd.DataFrame,
    output_dir: Path,
) -> None:
    for df, name in [
        (cluster_df_sc, "clusters_sc.csv"),
        (cluster_df_st, "clusters_st.csv"),
    ]:
        path = output_dir / name
        df.to_csv(path, index=False)
        logger.info(f"Saved cluster top-gene table → {path}")
