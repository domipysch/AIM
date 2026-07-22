"""Plots for analyzing a reference aligner's mapping_prob.h5ad output."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from metrics.onehot_plots import plot_dominance_thresholds, plot_onehot_distribution
from metrics.cossim_plots import plot_cossim_boxplots
from analysis.utils import run_pca_neighbors_umap

logger = logging.getLogger(__name__)

__all__ = [
    "plot_onehot_distribution",
    "plot_dominance_thresholds",
    "plot_cossim_boxplots",
    "build_celltype_palette",
    "plot_sc_umap_by_celltype",
    "plot_celltype_centroid_zscores",
    "plot_spatial_hard_celltypes",
]


def build_celltype_palette(cell_types: list[str]) -> dict[str, tuple]:
    """Consistent tab20 colour palette keyed by cell-type name, shared between
    the SC UMAP and the spatial hard-mapping plot so colours match."""
    cmap_base = plt.get_cmap("tab20")
    return {ct: cmap_base(i % 20) for i, ct in enumerate(cell_types)}


def plot_sc_umap_by_celltype(
    adata_sc: AnnData,
    cell_type_key: str,
    cell_types: list[str],
    palette: dict[str, tuple],
    output_path: Path,
) -> None:
    """PCA/neighbors/UMAP on a copy of the raw sc data, colored by the given
    cell-type obs column using the shared `palette` (so colours match the
    spatial hard-mapping plot)."""
    adata = adata_sc.copy()
    adata.obs[cell_type_key] = pd.Categorical(
        adata.obs[cell_type_key].astype(str), categories=cell_types
    )
    adata.uns[f"{cell_type_key}_colors"] = [
        mcolors.to_hex(palette[ct]) for ct in cell_types
    ]
    run_pca_neighbors_umap(adata, skip_umap=False)
    fig = sc.pl.umap(
        adata,
        color=cell_type_key,
        title="scRNA cells — cell type annotation",
        show=False,
        return_fig=True,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SC UMAP by cell type → %s", output_path)


def plot_celltype_centroid_zscores(centroids: pd.DataFrame, output_path: Path) -> None:
    """
    Heatmap of cell-type centroids (rows) x genes (cols), z-scored per gene
    across cell types.

    `centroids` should already be restricted to the genes to display (e.g.
    top marker genes) — this does not do any gene selection itself.
    """
    mat = centroids.to_numpy()
    col_std = mat.std(axis=0)
    col_std[col_std == 0] = 1.0
    mat_z = (mat - mat.mean(axis=0)) / col_std

    n_types, n_genes = mat_z.shape
    fig, ax = plt.subplots(
        figsize=(max(10, n_genes * 0.22), max(3, n_types * 0.6 + 1.5))
    )
    im = ax.imshow(mat_z, aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(centroids.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(centroids.index, fontsize=9)
    ax.set_xlabel("Marker gene (top per cell type, union)", fontsize=10)
    ax.set_ylabel("Cell type", fontsize=10)
    ax.set_title("Cell-type centroid expression (z-scored per gene)", fontsize=12)
    fig.colorbar(im, ax=ax, label="z-score", fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cell-type centroid z-score heatmap → %s", output_path)


def plot_spatial_hard_celltypes(
    adata_st: AnnData,
    hard_labels: np.ndarray,
    cell_types: list[str],
    palette: dict[str, tuple],
    output_path: Path,
    dot_size: float = 8.0,
) -> None:
    """Scatter of ST spots in physical space, colored by the hard-mapped
    (argmax) cell type using the shared `palette` (so colours match the SC
    UMAP). Coordinates are read from adata_st.obsm['spatial']."""
    if "spatial" not in adata_st.obsm:
        logger.warning("adata_st has no obsm['spatial'] — skipping spatial plot.")
        return

    coords = np.asarray(adata_st.obsm["spatial"])

    fig, ax = plt.subplots(figsize=(7, 6))
    for ct in cell_types:
        mask = hard_labels == ct
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[palette[ct]],
            s=dot_size,
            linewidths=0,
            alpha=0.85,
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_title("Spatial distribution of hard-mapped cell types", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Spatial hard cell-type plot → %s", output_path)
