"""
Pre-alignment compatibility check for SC / ST dataset pairs.

Usage
-----
Programmatic:
    from src.alternative_idea.src.pre_check import run_pre_check
    results = run_pre_check(sc_adata, st_adata, output_dir=Path("pre_check_out"))

CLI:
    python -m src.alternative_idea.src.pre_check \
        --scdata sc.h5ad --stdata st.h5ad --output_dir out/pre_check

The function runs three steps in order:
  1. Summary  — per-gene table (CSV) + dataset-level stats (JSON)
  2. Plots    — diagnostic figures saved to output_dir/plots/
  3. Metric   — centroid cosine similarity + optional permutation test
"""

import json
import logging
from pathlib import Path
from typing import Any

from anndata import AnnData

from .metric import (
    centroid_cosine_sim,
    greedy_cosine_sim,
    leiden_labels,
    permutation_test,
    shared_expression,
    top_gene_jaccard,
    variance_rank_spearman,
    zscored_centroids,
)
from .plots import save_all_plots
from .report import generate_pre_check_report
from .summary import (
    compute_cluster_top_genes,
    compute_dataset_summary,
    compute_gene_table,
    save_cluster_tables,
    save_summary,
)

logger = logging.getLogger(__name__)


def run_pre_check(
    sc_adata: AnnData,
    st_adata: AnnData,
    output_dir: Path,
    leiden_resolution: float = 0.5,
    run_permutation_test: bool = False,
    n_permutations: int = 200,
) -> dict:
    """
    Run the full pre-alignment compatibility check and save all outputs.

    Parameters
    ----------
    sc_adata, st_adata       : AnnData — shared genes are detected automatically
    output_dir               : directory to write summary, gene table, and plots
    leiden_resolution        : Leiden resolution for clustering both modalities
    run_permutation_test     : if True, validate metric against gene-shuffle null
    n_permutations           : number of permutations for the null distribution

    Returns
    -------
    dict with keys:
        n_shared_genes, n_clusters_sc, n_clusters_st,
        centroid_cosine_sim, permutation_test (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Shared genes + expression matrices
    # ------------------------------------------------------------------
    logger.info("Computing shared genes ...")
    X_sc, X_st, shared = shared_expression(sc_adata, st_adata)
    logger.info(f"  {len(shared)} shared genes")

    # ------------------------------------------------------------------
    # Clustering (done once; reused by both plots and metric)
    # ------------------------------------------------------------------
    logger.info(f"Clustering SC (resolution={leiden_resolution}) ...")
    labels_sc = leiden_labels(X_sc, leiden_resolution)
    n_clusters_sc = int(len(set(labels_sc)))
    logger.info(f"  SC: {n_clusters_sc} clusters")

    logger.info(f"Clustering ST (resolution={leiden_resolution}) ...")
    labels_st = leiden_labels(X_st, leiden_resolution)
    n_clusters_st = int(len(set(labels_st)))
    logger.info(f"  ST: {n_clusters_st} clusters")

    # ------------------------------------------------------------------
    # Step 1: Summary
    # ------------------------------------------------------------------
    logger.info("Computing data summary ...")
    dataset_summary = compute_dataset_summary(
        sc_adata,
        st_adata,
        X_sc,
        X_st,
        labels_sc,
        labels_st,
        shared,
        n_clusters_sc,
        n_clusters_st,
    )
    gene_table = compute_gene_table(X_sc, X_st, shared)
    save_summary(dataset_summary, gene_table, output_dir)

    cluster_df_sc = compute_cluster_top_genes(X_sc, labels_sc, shared, top_k=20)
    cluster_df_st = compute_cluster_top_genes(X_st, labels_st, shared, top_k=20)
    save_cluster_tables(cluster_df_sc, cluster_df_st, output_dir)

    # ------------------------------------------------------------------
    # Step 2: Plots
    # ------------------------------------------------------------------
    logger.info("Generating plots ...")
    C_sc = zscored_centroids(X_sc, labels_sc)
    C_st = zscored_centroids(X_st, labels_st)

    rho = variance_rank_spearman(X_sc, X_st)
    logger.info(f"Variance rank Spearman: {rho:.4f}")

    score, sim_matrix = centroid_cosine_sim(C_sc, C_st)
    greedy_score, greedy_sim, best_sc_per_st = greedy_cosine_sim(C_sc, C_st)
    logger.info(f"Greedy best-match cosine sim: {greedy_score:.4f}")

    perm_results = None
    if run_permutation_test:
        logger.info(f"Running permutation test ({n_permutations} permutations) ...")
        perm_results = permutation_test(C_sc, C_st, X_st, labels_st, n_permutations)

    save_all_plots(
        sc_adata=sc_adata,
        st_adata=st_adata,
        gene_table=gene_table,
        X_sc=X_sc,
        labels_sc=labels_sc,
        X_st=X_st,
        labels_st=labels_st,
        shared=shared,
        sim_matrix=sim_matrix,
        perm_results=perm_results,
        greedy_data={
            "sim_matrix": greedy_sim,
            "best_sc_per_st": best_sc_per_st,
            "hungarian_score": score,
            "greedy_score": greedy_score,
        },
        output_dir=output_dir,
        dataset_summary=dataset_summary,
    )

    # ------------------------------------------------------------------
    # Step 3: Metric report
    # ------------------------------------------------------------------
    jaccard_5 = top_gene_jaccard(cluster_df_sc, cluster_df_st, top_k=5)
    jaccard_10 = top_gene_jaccard(cluster_df_sc, cluster_df_st, top_k=10)
    jaccard_20 = top_gene_jaccard(cluster_df_sc, cluster_df_st, top_k=20)

    results: dict[str, Any] = {
        "n_shared_genes": len(shared),
        "n_clusters_sc": n_clusters_sc,
        "n_clusters_st": n_clusters_st,
        "variance_rank_spearman": rho,
        "centroid_cosine_sim": round(score, 4),
        "greedy_cosine_sim": greedy_score,
        "top_gene_jaccard_top5": jaccard_5,
        "top_gene_jaccard_top10": jaccard_10,
        "top_gene_jaccard_top20": jaccard_20,
    }
    if perm_results is not None:
        results["permutation_test"] = {
            k: v for k, v in perm_results.items() if k != "null_scores"
        }

    report_path = output_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_report(results)
    logger.info(f"Results saved → {report_path}")

    generate_pre_check_report(output_dir)

    return results


def _print_report(results: dict) -> None:
    print("\n=== Pre-Alignment Compatibility Check ===")
    print(f"  Shared genes              : {results['n_shared_genes']}")
    print(f"  SC clusters               : {results['n_clusters_sc']}")
    print(f"  ST clusters               : {results['n_clusters_st']}")
    print(f"  Variance rank Spearman    : {results['variance_rank_spearman']:.4f}")
    print(f"  Centroid cosine sim       : {results['centroid_cosine_sim']:.4f}")
    print(f"  Greedy cosine sim         : {results['greedy_cosine_sim']:.4f}")
    if "permutation_test" in results:
        pt = results["permutation_test"]
        print(
            f"  Permutation test          : z={pt['z_score']:.2f},  p={pt['p_value']:.3f}"
        )
    print(f"  Top-gene Jaccard (top  5) : {results['top_gene_jaccard_top5']:.4f}")
    print(f"  Top-gene Jaccard (top 10) : {results['top_gene_jaccard_top10']:.4f}")
    print(f"  Top-gene Jaccard (top 20) : {results['top_gene_jaccard_top20']:.4f}")
    print("=========================================\n")
