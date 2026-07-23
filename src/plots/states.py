"""Visualisation functions: UMAP comparisons, crosstab heatmap, state profiles."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from adata_schema import (
    OBS_COMPUTED_STATE,
    OBS_LEIDEN_ALL_GENES,
    OBS_LEIDEN_SHARED_GENES,
    OBSM_UMAP,
    OBSM_UMAP_SHARED_GENES,
    UNS_SHARED_GENES,
)
from analysis.utils import to_dense

logger = logging.getLogger(__name__)


def _build_state_palette(unique_states: list[int]) -> dict[int, tuple]:
    """Consistent tab20 colour palette keyed by integer state id."""
    cmap_base = plt.get_cmap("tab20")
    return {s: cmap_base(i % 20) for i, s in enumerate(unique_states)}


def _assign_computed_state_colors(
    adata: AnnData,
    state_palette: dict[int, tuple] | None,
) -> None:
    """Pre-assign adata.uns['computed_state_colors'] so scanpy uses our palette."""
    if state_palette is None or OBS_COMPUTED_STATE not in adata.obs:
        return
    cats = adata.obs[OBS_COMPUTED_STATE].cat.categories.tolist()
    adata.uns[f"{OBS_COMPUTED_STATE}_colors"] = [
        mcolors.to_hex(state_palette.get(int(c), (0.7, 0.7, 0.7, 1.0))) for c in cats
    ]


def _computed_state_count_line(adata: AnnData) -> str:
    """Annotation like '9 states' for the computed_state panel."""
    n_total = int(adata.obs[OBS_COMPUTED_STATE].nunique())
    return f"{n_total} states"


def plot_umap_comparison(
    adata: AnnData,
    panels: list[tuple[str, str]],
    output_path: Path,
    state_palette: dict[int, tuple] | None = None,
) -> None:
    """
    Save multiple UMAP panels side by side for visual comparison.

    Parameters
    ----------
    panels           : list of (color_key, title) pairs — all keys must exist in adata.obs.
    state_palette    : if provided, pre-assigned to adata.uns for the computed_state panel
                       so colours stay consistent with the spatial plot.
    """
    _assign_computed_state_colors(adata, state_palette)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (color_key, title) in zip(axes, panels):
        if color_key == OBS_COMPUTED_STATE:
            count_line = _computed_state_count_line(adata)
        else:
            count_line = f"{int(adata.obs[color_key].nunique())} clusters"
        sc.pl.umap(
            adata,
            color=color_key,
            title=f"{title}\n{count_line}",
            ax=ax,
            show=False,
            save=False,
            legend_loc=None,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP comparison → %s", output_path)


def plot_umap_grid(
    adata: AnnData,
    output_path: Path,
    leiden_resolution: float,
    state_palette: dict[int, tuple] | None = None,
    modularity_all: float | None = None,
    modularity_shared: float | None = None,
) -> None:
    """
    2x2 UMAP grid comparing the Leiden overclustering and the computed AIM
    states, on the all-gene embedding, plus the computed states on the
    shared-gene embedding:

                        Leiden (all-gene      Leiden (shared-gene
                        clustering)           clustering)
        all-gene UMAP   [leiden_state]        [leiden_shared_state]
                        Computed AIM states   Computed AIM states
                        (all genes)           (shared genes)
                        [computed_state]      [computed_state] (on
                        (+mod_all)            shared-gene UMAP, +mod_shared)

    ``adata`` must carry obs['leiden_state'], obs['leiden_shared_state'],
    obs['computed_state'], obsm[OBSM_UMAP] (all-gene) and
    obsm[OBSM_UMAP_SHARED_GENES] (shared-gene) — both embeddings live on the
    same object, so only the bottom-right panel switches basis. `modularity_all`
    / `modularity_shared`, if given, annotate the matching computed-state panel
    — each is the modularity of the computed partition on the same KNN graph
    its UMAP is projected from.
    """

    def _mod(m: float | None) -> str:
        return f"\nmodularity = {m:.3f}" if m is not None and m == m else ""

    _assign_computed_state_colors(adata, state_palette)

    res = leiden_resolution
    # (row, col, basis, obs key, base title, caption suffix)
    panels = [
        (
            0,
            0,
            OBSM_UMAP,
            OBS_LEIDEN_ALL_GENES,
            f"Leiden overclusters — all genes (resolution={res})",
            "",
        ),
        (
            0,
            1,
            OBSM_UMAP,
            OBS_LEIDEN_SHARED_GENES,
            f"Leiden shared-gene clusters — all-gene UMAP (resolution={res})",
            "",
        ),
        (
            1,
            0,
            OBSM_UMAP,
            OBS_COMPUTED_STATE,
            "Computed AIM states — all genes",
            _mod(modularity_all),
        ),
        (
            1,
            1,
            OBSM_UMAP_SHARED_GENES,
            OBS_COMPUTED_STATE,
            "Computed AIM states — shared genes (ST overlap)",
            _mod(modularity_shared),
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for r, c, basis, key, title, suffix in panels:
        ax = axes[r][c]
        if key == OBS_COMPUTED_STATE:
            count_line = _computed_state_count_line(adata)
        else:
            count_line = f"{int(adata.obs[key].nunique())} clusters"
        sc.pl.embedding(
            adata,
            basis=basis,
            color=key,
            title=f"{title}\n{count_line}{suffix}",
            ax=ax,
            show=False,
            save=False,
            legend_loc=None,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP grid → %s", output_path)


def plot_leiden_merge_map(
    leiden_labels: np.ndarray,
    cell_states: np.ndarray,
    output_path: Path,
    state_palette: dict[int, tuple] | None = None,
) -> None:
    """
    Visualise which Leiden overclusters were merged into each computed AIM state.

    One horizontal bar per computed state, split into segments — one per Leiden
    cluster merged into that state — each segment's width proportional to that
    cluster's cell count and labelled with its Leiden id. A state built from a
    single Leiden cluster shows one segment; a merged state shows several, so
    the merge structure (and the relative sizes of what was merged) is read off
    directly.
    """
    leiden_labels = np.asarray(leiden_labels)
    cell_states = np.asarray(cell_states)

    states = sorted(np.unique(cell_states).tolist())
    # Each Leiden cluster maps to exactly one state (state = argmax of the merge
    # matrix G for that cluster), so cell_states is constant within a cluster.
    leiden_of_state: dict[int, list[tuple[int, int]]] = {s: [] for s in states}
    for lc in np.unique(leiden_labels):
        mask = leiden_labels == lc
        s = int(cell_states[mask][0])
        leiden_of_state[s].append((int(lc), int(mask.sum())))
    for s in states:
        leiden_of_state[s].sort(key=lambda t: t[1], reverse=True)

    n_states = len(states)
    fig, ax = plt.subplots(figsize=(11, max(3, n_states * 0.5 + 1.5)))
    for row, s in enumerate(states):
        color = (
            state_palette.get(s, (0.6, 0.6, 0.6, 1.0))
            if state_palette is not None
            else (0.6, 0.6, 0.6, 1.0)
        )
        x = 0.0
        for lc, size in leiden_of_state[s]:
            ax.barh(row, size, left=x, color=color, edgecolor="white", linewidth=1.5)
            ax.text(
                x + size / 2,
                row,
                f"L{lc}",
                va="center",
                ha="center",
                fontsize=7,
                color="black",
            )
            x += size

    ax.set_yticks(range(n_states))
    ax.set_yticklabels(
        [
            f"State {s}  ({len(leiden_of_state[s])} "
            f"cluster{'s' if len(leiden_of_state[s]) != 1 else ''})"
            for s in states
        ],
        fontsize=9,
    )
    ax.set_xlabel(
        "cells   (bar = one AIM state; segments = merged Leiden clusters, "
        "width proportional to cluster size)",
        fontsize=10,
    )
    ax.set_title("Leiden overclusters merged per computed AIM state", fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Leiden merge map → %s", output_path)


def plot_state_profiles(
    adata_sc: AnnData,
    cell_states: np.ndarray,
    output_path: Path,
    cell_fractions: dict[int, float] | None = None,
    spot_fractions: dict[int, float] | None = None,
    state_palette: dict[int, tuple] | None = None,
) -> None:
    """
    Cluster-mean expression heatmap for each computed cell state, with optional
    cell-fraction and spot-fraction bar charts on the right.

    Genes are restricted to shared_genes and sorted by SC variance (highest
    first).  Expression is z-scored per gene across states so that the colour
    encodes how distinctively each state expresses every gene.
    """
    shared_genes = list(adata_sc.uns[UNS_SHARED_GENES])
    available = [g for g in shared_genes if g in adata_sc.var_names]
    if len(available) < 2:
        logger.warning("Too few shared genes for state-profile plot — skipping.")
        return

    X = to_dense(adata_sc[:, available])
    gene_names = np.array(available)
    gene_order = np.argsort(X.var(axis=0))[::-1]

    unique_states = sorted(np.unique(cell_states))
    n_states = len(unique_states)

    mat = np.stack(
        [X[cell_states == k][:, gene_order].mean(axis=0) for k in unique_states]
    )
    col_std = mat.std(axis=0)
    col_std[col_std == 0] = 1.0
    mat_z = (mat - mat.mean(axis=0)) / col_std

    n_genes = len(available)
    n_panels = (
        1
        + (1 if cell_fractions is not None else 0)
        + (1 if spot_fractions is not None else 0)
    )
    width_ratios = [n_genes] + [max(3, n_genes // 12)] * (n_panels - 1)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            max(12, n_genes * 0.18) + 3 * (n_panels - 1),
            max(3, n_states * 0.5 + 2),
        ),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    ax_heat = axes[0]
    im = ax_heat.imshow(mat_z, aspect="auto", cmap="viridis")
    ax_heat.set_xticks(range(n_genes))
    ax_heat.set_xticklabels(gene_names[gene_order], rotation=90, fontsize=5)
    ax_heat.set_yticks(range(n_states))
    ax_heat.set_yticklabels([f"State {k}" for k in unique_states], fontsize=8)
    ax_heat.set_xlabel(
        "Gene  (sorted by SC variance, z-scored across states)", fontsize=10
    )
    ax_heat.set_ylabel("Computed cell state", fontsize=10)
    ax_heat.set_title("Cell-state profiles — shared genes", fontsize=12)
    fig.colorbar(im, ax=ax_heat, label="z-score", fraction=0.015, pad=0.01)

    ax_idx = 1
    for fractions, label, fallback_color in [
        (cell_fractions, "Cell fraction", "steelblue"),
        (spot_fractions, "Spot fraction", "darkorange"),
    ]:
        if fractions is None:
            continue
        ax = axes[ax_idx]
        values = [fractions.get(k, 0.0) for k in unique_states]
        bar_colors = (
            [state_palette.get(k, (0.7, 0.7, 0.7, 1.0)) for k in unique_states]
            if state_palette is not None
            else fallback_color
        )
        ax.barh(range(n_states), values, color=bar_colors, alpha=0.8)
        for i, v in enumerate(values):
            ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=7)
        ax.set_yticks(range(n_states))
        ax.set_yticklabels([])
        ax.set_xlabel(label, fontsize=9)
        ax.set_xlim(0, max(values) * 1.3 if max(values) > 0 else 0.1)
        ax.set_title(label, fontsize=10)
        ax.invert_yaxis()
        ax_idx += 1

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("State profiles → %s", output_path)


def plot_state_fractions(
    cell_fractions: dict[int, float],
    spot_fractions: dict[int, float],
    unique_states: list[int],
    output_path: Path,
    state_palette: dict[int, tuple] | None = None,
) -> None:
    """
    Standalone export of the cell-fraction and spot-fraction bar charts.

    Produces the same bar plots that appear on the right side of
    cell_state_profiles.png, but as a self-contained image with proper
    y-axis state labels.
    """
    panels = [
        (cell_fractions, "Cell fraction", "steelblue"),
        (spot_fractions, "Spot fraction", "darkorange"),
    ]
    active = [(f, lbl, c) for f, lbl, c in panels if f]
    if not active:
        return

    n_states = len(unique_states)
    fig, axes = plt.subplots(
        1,
        len(active),
        figsize=(3.5 * len(active), max(3, n_states * 0.45 + 1.5)),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (fractions, label, fallback_color) in zip(axes, active):
        values = [fractions.get(k, 0.0) for k in unique_states]
        bar_colors = (
            [state_palette.get(k, (0.7, 0.7, 0.7, 1.0)) for k in unique_states]
            if state_palette is not None
            else fallback_color
        )
        ax.barh(range(n_states), values, color=bar_colors, alpha=0.8)
        for i, v in enumerate(values):
            ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=8)
        ax.set_yticks(range(n_states))
        ax.set_yticklabels([f"State {k}" for k in unique_states], fontsize=8)
        ax.set_xlabel(label, fontsize=10)
        ax.set_xlim(0, max(values) * 1.35 if max(values) > 0 else 0.1)
        ax.set_title(label, fontsize=11)
        ax.invert_yaxis()

    fig.suptitle("Cell-state fractions", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("State fractions → %s", output_path)


def plot_spatial_cell_states(
    adata_st: AnnData,
    spot_states: np.ndarray,
    output_path: Path,
    dot_size: float = 8.0,
    state_palette: dict[int, tuple] | None = None,
) -> None:
    """
    Scatter plot of ST spots in physical space, coloured by computed cell-state.

    Coordinates are read from adata_st.obsm["spatial"]. state_palette syncs
    colours with the UMAP.
    """
    if "spatial" not in adata_st.obsm:
        logger.warning("adata_st has no obsm['spatial'] — skipping spatial plot.")
        return

    coords = np.asarray(adata_st.obsm["spatial"])
    unique_states = sorted(np.unique(spot_states).tolist())
    n_states = len(unique_states)

    # Use provided palette or fall back to tab20
    if state_palette is not None:
        state_to_color = {
            s: state_palette[s] for s in unique_states if s in state_palette
        }
    else:
        cmap_base = plt.get_cmap("tab20")
        state_to_color = {s: cmap_base(i % 20) for i, s in enumerate(unique_states)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for state, color in state_to_color.items():
        mask = spot_states == state
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            s=dot_size,
            label=f"State {state}",
            linewidths=0,
            alpha=0.85,
            rasterized=True,
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_title(
        f"Spatial distribution of computed cell states  ({n_states} states)",
        fontsize=12,
    )

    # Legend: compact when many states
    legend_fs = max(5, min(9, 120 // n_states))
    ncol = max(1, n_states // 20)
    ax.legend(
        loc="upper right",
        fontsize=legend_fs,
        markerscale=1.5,
        ncol=ncol,
        framealpha=0.7,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Spatial cell-state plot → %s", output_path)
