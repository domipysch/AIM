"""
Diagnostic plots for the pre-alignment compatibility check.

Each plot_* function saves a PNG to output_dir and closes the figure.
save_all_plots() is the single entry point called by __init__.py.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_FIG_DPI = 150
_CMAP_EXPR = "viridis"
_CMAP_SIM = "RdYlGn"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot → {path}")


# ---------------------------------------------------------------------------
# 1. Library size distributions
# ---------------------------------------------------------------------------


def plot_library_sizes(sc_adata: AnnData, st_adata: AnnData, output_dir: Path) -> None:
    """Histograms of total counts per cell (SC) and per spot (ST)."""
    from .metric import to_dense

    sc_libsize = to_dense(sc_adata.X).sum(axis=1)
    st_libsize = to_dense(st_adata.X).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, label, color in zip(
        axes,
        [sc_libsize, st_libsize],
        ["SC — total counts per cell", "ST — total counts per spot"],
        ["steelblue", "darkorange"],
    ):
        ax.hist(data, bins=60, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(
            np.median(data),
            color="black",
            linewidth=1.2,
            linestyle="--",
            label=f"median={np.median(data):.0f}",
        )
        ax.set_title(label)
        ax.set_xlabel("Total counts")
        ax.set_ylabel("# cells / spots")
        ax.legend(fontsize=8)

    fig.suptitle("Library size distributions", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "library_sizes.png")


# ---------------------------------------------------------------------------
# 2. Mean expression scatter (SC vs ST, log-scale)
# ---------------------------------------------------------------------------


def plot_mean_scatter(gene_table: pd.DataFrame, output_dir: Path) -> None:
    """Per-gene mean expression in SC vs ST (log scale). Diagonal = perfect match."""
    fig, ax = plt.subplots(figsize=(5, 5))

    x = gene_table["mean_sc"].values
    y = gene_table["mean_st"].values
    ax.scatter(x, y, s=6, alpha=0.5, linewidths=0, color="steelblue")

    # diagonal reference
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)

    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Mean expression SC")
    ax.set_ylabel("Mean expression ST")
    ax.set_title("Per-gene mean expression (shared genes)")
    fig.tight_layout()
    _save(fig, output_dir / "mean_scatter.png")


# ---------------------------------------------------------------------------
# 3. Variance rank scatter
# ---------------------------------------------------------------------------


def plot_variance_rank_scatter(gene_table: pd.DataFrame, output_dir: Path) -> None:
    """Variance rank in SC vs ST. Diagonal = same genes carry signal in both."""
    fig, ax = plt.subplots(figsize=(5, 5))

    x = gene_table["var_rank_sc"].values
    y = gene_table["var_rank_st"].values
    diff = gene_table["var_rank_diff"].values
    scatter = ax.scatter(x, y, c=diff, s=5, alpha=0.6, linewidths=0, cmap="plasma_r")
    fig.colorbar(scatter, ax=ax, label="|rank_sc − rank_st|")

    n = len(gene_table)
    ax.plot([1, n], [1, n], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Variance rank SC  (1 = most variable)")
    ax.set_ylabel("Variance rank ST  (1 = most variable)")
    ax.set_title("Variance rank concordance (shared genes)")
    fig.tight_layout()
    _save(fig, output_dir / "variance_rank_scatter.png")


# ---------------------------------------------------------------------------
# 4. Dropout scatter
# ---------------------------------------------------------------------------


def plot_dropout_scatter(gene_table: pd.DataFrame, output_dir: Path) -> None:
    """Fraction of zeros per gene in SC vs ST. Discordant genes are far from diagonal."""
    fig, ax = plt.subplots(figsize=(5, 5))

    x = gene_table["dropout_sc"].values
    y = gene_table["dropout_st"].values
    ax.scatter(x, y, s=6, alpha=0.5, linewidths=0, color="tomato")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Dropout rate SC (fraction zeros)")
    ax.set_ylabel("Dropout rate ST (fraction zeros)")
    ax.set_title("Per-gene dropout concordance (shared genes)")
    fig.tight_layout()
    _save(fig, output_dir / "dropout_scatter.png")


# ---------------------------------------------------------------------------
# 5. UMAP embeddings (SC + ST side by side)
# ---------------------------------------------------------------------------


def _compute_umap(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, Any]:
    adata = sc.AnnData(X=X.copy())
    old_verbosity = sc.settings.verbosity
    sc.settings.verbosity = 0
    try:
        sc.pp.neighbors(adata, use_rep="X", random_state=42)
        sc.tl.umap(adata, random_state=42)
    finally:
        sc.settings.verbosity = old_verbosity
    adata.obs["cluster"] = labels.astype(str)
    return adata.obsm["X_umap"], adata


def plot_umap(
    X_sc: np.ndarray,
    labels_sc: np.ndarray,
    X_st: np.ndarray,
    labels_st: np.ndarray,
    output_dir: Path,
    sil_sc: float | None = None,
    dunn_sc: float | None = None,
    sil_st: float | None = None,
    dunn_st: float | None = None,
) -> None:
    """UMAP of SC cells and ST spots, each colored by Leiden cluster."""

    def _quality_str(sil, dunn):
        parts = []
        if sil is not None:
            parts.append(f"silhouette={sil:.3f}")
        if dunn is not None:
            parts.append(f"Dunn={dunn:.3f}")
        return f"  ({', '.join(parts)})" if parts else ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, X, labels, base_title, sil, dunn in zip(
        axes,
        [X_sc, X_st],
        [labels_sc, labels_st],
        ["SC — Leiden clusters", "ST — Leiden clusters"],
        [sil_sc, sil_st],
        [dunn_sc, dunn_st],
    ):
        title = base_title + _quality_str(sil, dunn)
        try:
            umap_coords, _ = _compute_umap(X, labels)
            unique_labels = sorted(np.unique(labels))
            cmap = plt.colormaps.get_cmap("tab20")
            for i, lab in enumerate(unique_labels):
                mask = labels == lab
                ax.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    s=4,
                    alpha=0.6,
                    linewidths=0,
                    color=cmap(i % 20),
                    label=str(lab),
                )
            ax.set_title(title)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(
                markerscale=2, fontsize=7, loc="best", title="Cluster", title_fontsize=7
            )
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"UMAP failed:\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title(title)

    fig.tight_layout()
    _save(fig, output_dir / "umap.png")


# ---------------------------------------------------------------------------
# 6. Spatial plot of ST clusters
# ---------------------------------------------------------------------------


def plot_spatial(st_adata: AnnData, labels_st: np.ndarray, output_dir: Path) -> None:
    """ST spots on spatial coordinates, colored by Leiden cluster."""
    if "spatial" not in st_adata.obsm:
        logger.warning("st_adata has no obsm['spatial'] — skipping spatial plot.")
        return

    coords = st_adata.obsm["spatial"]
    unique_labels = sorted(np.unique(labels_st))
    cmap = plt.colormaps.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, lab in enumerate(unique_labels):
        mask = labels_st == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=0.7,
            linewidths=0,
            color=cmap(i % 20),
            label=str(lab),
        )
    ax.set_title("ST — Leiden clusters on spatial coordinates")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(markerscale=2, fontsize=7, loc="best", title="Cluster", title_fontsize=7)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, output_dir / "spatial_clusters.png")


# ---------------------------------------------------------------------------
# 7. Centroid cosine similarity heatmap (full matrix)
# ---------------------------------------------------------------------------


def plot_centroid_heatmap(
    sim_matrix: np.ndarray,
    output_dir: Path,
    labels_sc: np.ndarray | None = None,
    labels_st: np.ndarray | None = None,
    hungarian_score: float | None = None,
) -> None:
    """Full SC-cluster × ST-cluster cosine similarity matrix as a heatmap."""
    from scipy.optimize import linear_sum_assignment

    row_idx, col_idx = linear_sum_assignment(-sim_matrix)
    if hungarian_score is None:
        hungarian_score = float(sim_matrix[row_idx, col_idx].mean())

    def _tick_labels(labels: np.ndarray | None, n: int, prefix: str) -> list[str]:
        if labels is None:
            return [f"{prefix}{k}" for k in range(n)]
        unique = sorted(np.unique(labels))
        total = len(labels)
        return [f"{prefix}{k}\n{(labels == k).sum() / total:.1%}" for k in unique]

    sc_ticks = _tick_labels(labels_sc, sim_matrix.shape[0], "C")
    st_ticks = _tick_labels(labels_st, sim_matrix.shape[1], "C")

    fig, ax = plt.subplots(
        figsize=(
            max(5, sim_matrix.shape[1] * 0.6 + 1),
            max(4, sim_matrix.shape[0] * 0.6 + 1),
        )
    )
    im = ax.imshow(sim_matrix, aspect="auto", cmap=_CMAP_SIM, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Cosine similarity")

    ax.scatter(
        col_idx,
        row_idx,
        marker="*",
        s=60,
        color="black",
        zorder=5,
        label=f"Hungarian match  (mean={hungarian_score:.3f})",
    )
    ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("ST cluster")
    ax.set_ylabel("SC cluster")
    ax.set_title("Centroid cosine similarity (SC vs ST clusters)")
    ax.set_xticks(range(sim_matrix.shape[1]))
    ax.set_xticklabels(st_ticks, fontsize=7)
    ax.set_yticks(range(sim_matrix.shape[0]))
    ax.set_yticklabels(sc_ticks, fontsize=7)
    fig.tight_layout()
    _save(fig, output_dir / "centroid_heatmap.png")


# ---------------------------------------------------------------------------
# 7b. Greedy best-match heatmap
# ---------------------------------------------------------------------------


def plot_greedy_heatmap(
    sim_matrix: np.ndarray,
    best_sc_per_st: np.ndarray,
    output_dir: Path,
    labels_sc: np.ndarray | None = None,
    labels_st: np.ndarray | None = None,
    greedy_score: float | None = None,
) -> None:
    """
    Same layout as the Hungarian heatmap, but marks each ST cluster's best SC
    match independently (many-to-one allowed).
    """
    if greedy_score is None:
        st_indices = np.arange(len(best_sc_per_st))
        greedy_score = float(sim_matrix[best_sc_per_st, st_indices].mean())

    def _tick_labels(labels: np.ndarray | None, n: int, prefix: str) -> list[str]:
        if labels is None:
            return [f"{prefix}{k}" for k in range(n)]
        unique = sorted(np.unique(labels))
        total = len(labels)
        return [f"{prefix}{k}\n{(labels == k).sum() / total:.1%}" for k in unique]

    sc_ticks = _tick_labels(labels_sc, sim_matrix.shape[0], "C")
    st_ticks = _tick_labels(labels_st, sim_matrix.shape[1], "C")

    fig, ax = plt.subplots(
        figsize=(
            max(5, sim_matrix.shape[1] * 0.6 + 1),
            max(4, sim_matrix.shape[0] * 0.6 + 1),
        )
    )
    im = ax.imshow(sim_matrix, aspect="auto", cmap=_CMAP_SIM, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Cosine similarity")

    st_indices = np.arange(len(best_sc_per_st))
    ax.scatter(
        st_indices,
        best_sc_per_st,
        marker="*",
        s=60,
        color="black",
        zorder=5,
        label=f"Greedy best match  (mean={greedy_score:.3f})",
    )
    ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("ST cluster")
    ax.set_ylabel("SC cluster")
    ax.set_title("Greedy best-match cosine similarity (SC vs ST clusters)")
    ax.set_xticks(range(sim_matrix.shape[1]))
    ax.set_xticklabels(st_ticks, fontsize=7)
    ax.set_yticks(range(sim_matrix.shape[0]))
    ax.set_yticklabels(sc_ticks, fontsize=7)
    fig.tight_layout()
    _save(fig, output_dir / "centroid_heatmap_greedy.png")


# ---------------------------------------------------------------------------
# 8. Permutation test null distribution
# ---------------------------------------------------------------------------


def plot_permutation_null(perm_results: dict, output_dir: Path) -> None:
    """Histogram of null scores with the real score marked."""
    null_scores = np.array(perm_results["null_scores"])
    real_score = perm_results["real_score"]
    p_value = perm_results["p_value"]
    z_score = perm_results["z_score"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        null_scores,
        bins=40,
        color="lightgray",
        edgecolor="white",
        label=f"Null (n={len(null_scores)})",
    )
    ax.axvline(
        real_score, color="crimson", linewidth=2, label=f"Real score = {real_score:.3f}"
    )
    ax.axvline(
        perm_results["null_mean"],
        color="dimgray",
        linewidth=1.2,
        linestyle="--",
        label=f"Null mean = {perm_results['null_mean']:.3f}",
    )

    ax.set_xlabel("Centroid cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation test   z = {z_score:.2f},  p = {p_value:.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir / "permutation_null.png")


# ---------------------------------------------------------------------------
# 9. HVG expression heatmap per modality (same gene order)
# ---------------------------------------------------------------------------


def plot_hvg_heatmap(
    X_sc: np.ndarray,
    labels_sc: np.ndarray,
    X_st: np.ndarray,
    labels_st: np.ndarray,
    gene_names: list[str],
    output_dir: Path,
) -> None:
    """
    Cluster-mean expression heatmap for SC and ST using all shared genes in the
    same order (sorted by SC variance, highest first). Both rows share the same
    gene axis, making cross-modality comparison direct.
    """
    gene_arr = np.array(gene_names)
    n_genes = len(gene_arr)

    # shared gene order: sort by SC variance descending
    gene_order = np.argsort(X_sc.var(axis=0))[::-1]

    def _cluster_mean_row(X, labels):
        unique = sorted(np.unique(labels))
        mat = np.stack([X[labels == k][:, gene_order].mean(axis=0) for k in unique])
        col_std = mat.std(axis=0)
        col_std[col_std == 0] = 1.0
        return (mat - mat.mean(axis=0)) / col_std, unique

    mat_sc, unique_sc = _cluster_mean_row(X_sc, labels_sc)
    mat_st, unique_st = _cluster_mean_row(X_st, labels_st)

    fig, axes = plt.subplots(
        2, 1, figsize=(max(12, n_genes * 0.18), 8), gridspec_kw={"hspace": 0.55}
    )

    for ax, mat, unique, title in zip(
        axes,
        [mat_sc, mat_st],
        [unique_sc, unique_st],
        [
            "SC — all shared genes (sorted by SC variance)",
            "ST — all shared genes (same order)",
        ],
    ):
        im = ax.imshow(mat, aspect="auto", cmap=_CMAP_EXPR)
        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(gene_arr[gene_order], rotation=90, fontsize=5)
        ax.set_yticks(range(len(unique)))
        ax.set_yticklabels([f"C{k}" for k in unique], fontsize=7)
        ax.set_xlabel("Gene (z-scored mean per cluster, same order in both)")
        ax.set_ylabel("Cluster")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="z-score", fraction=0.02, pad=0.01)

    fig.suptitle("Cluster-mean expression — shared genes", fontsize=11)
    _save(fig, output_dir / "hvg_heatmap.png")


# ---------------------------------------------------------------------------
# 10. Per-match cluster heatmaps
# ---------------------------------------------------------------------------


def plot_matched_cluster_heatmaps(
    X_sc: np.ndarray,
    labels_sc: np.ndarray,
    X_st: np.ndarray,
    labels_st: np.ndarray,
    gene_names: list[str],
    sim_matrix: np.ndarray,
    output_dir: Path,
) -> None:
    """
    For every Hungarian-matched SC–ST cluster pair, save a figure with two rows:
      top    — mean expression of SC cells in that cluster
      bottom — mean expression of ST spots in the matched cluster
    Genes are in the same order (sorted by SC cluster expression descending),
    z-scored per row for visual contrast. One PNG per match in matched_clusters/.
    """
    from scipy.optimize import linear_sum_assignment

    row_idx, col_idx = linear_sum_assignment(-sim_matrix)
    gene_arr = np.array(gene_names)
    n_genes = len(gene_arr)

    # same gene order as hvg_heatmap: sorted by global SC variance descending
    gene_order = np.argsort(X_sc.var(axis=0))[::-1]

    match_dir = output_dir / "matched_clusters"
    match_dir.mkdir(parents=True, exist_ok=True)

    def _zrow(v):
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v - v.mean()

    n_matches = len(row_idx)
    all_mats = []

    for sc_k, st_k in zip(row_idx, col_idx):
        sc_centroid = X_sc[labels_sc == sc_k].mean(axis=0)
        st_centroid = X_st[labels_st == st_k].mean(axis=0)
        mat = np.stack([_zrow(sc_centroid[gene_order]), _zrow(st_centroid[gene_order])])
        all_mats.append((sc_k, st_k, mat))

        # individual plot
        cos_sim = float(sim_matrix[sc_k, st_k])
        fig, ax = plt.subplots(figsize=(max(12, n_genes * 0.18), 3))
        im = ax.imshow(mat, aspect="auto", cmap=_CMAP_EXPR, vmin=-3, vmax=3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"SC C{sc_k}", f"ST C{st_k}"], fontsize=9)
        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(gene_arr[gene_order], rotation=90, fontsize=5)
        ax.set_xlabel("Gene (sorted by SC variance, z-scored per row)")
        ax.set_title(
            f"Match: SC cluster {sc_k}  ↔  ST cluster {st_k}   (cosine sim = {cos_sim:.3f})"
        )
        fig.colorbar(im, ax=ax, label="z-score", fraction=0.02, pad=0.01)
        fig.tight_layout()
        _save(fig, match_dir / f"match_SC{sc_k}_ST{st_k}.png")

    # combined overview: all matches stacked vertically
    row_h = 2.2
    fig, axes = plt.subplots(
        n_matches,
        1,
        figsize=(max(12, n_genes * 0.18), row_h * n_matches),
        gridspec_kw={"hspace": 0.8},
    )
    if n_matches == 1:
        axes = [axes]

    for ax, (sc_k, st_k, mat) in zip(axes, all_mats):
        cos_sim = float(sim_matrix[sc_k, st_k])
        im = ax.imshow(mat, aspect="auto", cmap=_CMAP_EXPR, vmin=-3, vmax=3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"SC C{sc_k}", f"ST C{st_k}"], fontsize=7)
        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(gene_arr[gene_order], rotation=90, fontsize=5)
        ax.set_title(
            f"SC {sc_k} ↔ ST {st_k}  (cos={cos_sim:.3f})",
            fontsize=8,
            pad=3,
        )
        fig.colorbar(im, ax=ax, label="z", fraction=0.01, pad=0.005)

    fig.suptitle(
        "All matched clusters — gene expression overview", fontsize=11, y=1.001
    )
    _save(fig, match_dir / "all_matches_overview.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def save_all_plots(
    sc_adata: AnnData,
    st_adata: AnnData,
    gene_table: pd.DataFrame,
    X_sc: np.ndarray,
    labels_sc: np.ndarray,
    X_st: np.ndarray,
    labels_st: np.ndarray,
    shared: list[str],
    sim_matrix: np.ndarray,
    perm_results: dict | None,
    greedy_data: dict | None,
    output_dir: Path,
    dataset_summary: dict | None = None,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sil_sc = dataset_summary.get("silhouette_sc") if dataset_summary else None
    dunn_sc = dataset_summary.get("dunn_sc") if dataset_summary else None
    sil_st = dataset_summary.get("silhouette_st") if dataset_summary else None
    dunn_st = dataset_summary.get("dunn_st") if dataset_summary else None

    plot_library_sizes(sc_adata, st_adata, plots_dir)
    plot_mean_scatter(gene_table, plots_dir)
    plot_variance_rank_scatter(gene_table, plots_dir)
    plot_dropout_scatter(gene_table, plots_dir)
    plot_umap(
        X_sc,
        labels_sc,
        X_st,
        labels_st,
        plots_dir,
        sil_sc=sil_sc,
        dunn_sc=dunn_sc,
        sil_st=sil_st,
        dunn_st=dunn_st,
    )
    plot_spatial(st_adata, labels_st, plots_dir)
    plot_centroid_heatmap(
        sim_matrix,
        plots_dir,
        labels_sc=labels_sc,
        labels_st=labels_st,
        hungarian_score=greedy_data.get("hungarian_score") if greedy_data else None,
    )
    if greedy_data is not None:
        plot_greedy_heatmap(
            greedy_data["sim_matrix"],
            greedy_data["best_sc_per_st"],
            plots_dir,
            labels_sc=labels_sc,
            labels_st=labels_st,
            greedy_score=greedy_data.get("greedy_score"),
        )
    if perm_results is not None:
        plot_permutation_null(perm_results, plots_dir)
    plot_hvg_heatmap(X_sc, labels_sc, X_st, labels_st, shared, plots_dir)
    plot_matched_cluster_heatmaps(
        X_sc, labels_sc, X_st, labels_st, shared, sim_matrix, plots_dir
    )

    logger.info(f"All plots saved to {plots_dir}")
