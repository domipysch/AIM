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

from utils import _dense_X

logger = logging.getLogger(__name__)

_FRAC_THRESHOLD = 0.01  # states with cell- or spot-fraction below this are hidden


def _build_state_palette(unique_states: list[int]) -> dict[int, tuple]:
    """Consistent tab20 colour palette keyed by integer state id."""
    cmap_base = plt.get_cmap("tab20")
    return {s: cmap_base(i % 20) for i, s in enumerate(unique_states)}


def plot_umap(
    adata: AnnData,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    color_key: str = "_cluster",
) -> None:
    """Add labels to adata.obs[color_key] and save a UMAP figure."""
    adata.obs[color_key] = pd.Categorical(labels.astype(str))
    fig = sc.pl.umap(adata, color=color_key, title=title, show=False, return_fig=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP → %s", output_path)


def plot_umap_comparison(
    adata: AnnData,
    panels: list[tuple[str, str]],
    output_path: Path,
    cell_fractions: dict[int, float] | None = None,
    spot_fractions: dict[int, float] | None = None,
    state_palette: dict[int, tuple] | None = None,
) -> None:
    """
    Save multiple UMAP panels side by side for visual comparison.

    Parameters
    ----------
    panels           : list of (color_key, title) pairs — all keys must exist in adata.obs.
    cell_fractions   : if provided, used to annotate the computed_state panel with
                       the number of states above _FRAC_THRESHOLD.
    spot_fractions   : same purpose as cell_fractions.
    state_palette    : if provided, pre-assigned to adata.uns for the computed_state panel
                       so colours stay consistent with the spatial plot.
    """
    # Pre-assign palette so scanpy uses our colours for computed_state
    if state_palette is not None and "computed_state" in adata.obs:
        cats = adata.obs["computed_state"].cat.categories.tolist()
        adata.uns["computed_state_colors"] = [
            mcolors.to_hex(state_palette.get(int(c), (0.7, 0.7, 0.7, 1.0)))
            for c in cats
        ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (color_key, title) in zip(axes, panels):
        n_total = int(adata.obs[color_key].nunique())
        if (
            color_key == "computed_state"
            and cell_fractions is not None
            and spot_fractions is not None
        ):
            n_above = sum(
                1
                for k, cf in cell_fractions.items()
                if cf > _FRAC_THRESHOLD and spot_fractions.get(k, 0.0) > _FRAC_THRESHOLD
            )
            count_line = f"{n_total} states  ({n_above} > 1 %)"
        else:
            count_line = f"{n_total} clusters"
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


def plot_crosstab_heatmap(
    cell_states: np.ndarray,
    gt_labels: np.ndarray,
    output_path: Path,
    gt_label_name: str = "GT label",
) -> None:
    """Log-scale heatmap of raw crosstab counts: predicted state × GT cell type."""
    crosstab = pd.crosstab(
        pd.Series(cell_states, name="predLabel"),
        pd.Series(gt_labels, name=gt_label_name),
    )
    data = crosstab.values.astype(float)
    total = data.sum()
    row_counts = data.sum(axis=1)
    col_counts = data.sum(axis=0)

    n_rows, n_cols = data.shape
    y_labels = [
        f"State {s}  n={int(n)}  ({n / total * 100:.1f}%)"
        for s, n in zip(crosstab.index.tolist(), row_counts)
    ]
    x_labels = [
        f"{gt}\nn={int(n)}  ({n / total * 100:.1f}%)"
        for gt, n in zip(crosstab.columns.tolist(), col_counts)
    ]

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.7), max(5, n_rows * 0.6)))
    pos_vals = data[data > 0]
    vmin = float(pos_vals.min()) if len(pos_vals) else 1.0
    norm = mcolors.LogNorm(vmin=vmin, vmax=max(float(data.max()), vmin))
    im = ax.imshow(data, cmap="YlOrRd", norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, label="Cell count (log scale)")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel(gt_label_name, fontsize=12)
    ax.set_ylabel("Predicted state", fontsize=12)
    ax.set_title(
        f"Crosstab: cell counts per predicted state × {gt_label_name}", fontsize=13
    )

    threshold = data.max() * 0.5
    for i in range(n_rows):
        for j in range(n_cols):
            val = int(data[i, j])
            if val == 0:
                continue
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if data[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Crosstab heatmap → %s", output_path)


def plot_state_profiles(
    adata_sc: AnnData,
    cell_states: np.ndarray,
    shared_genes: list[str],
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
    available = [g for g in shared_genes if g in adata_sc.var_names]
    if len(available) < 2:
        logger.warning("Too few shared genes for state-profile plot — skipping.")
        return

    X = _dense_X(adata_sc[:, available])
    gene_names = np.array(available)
    gene_order = np.argsort(X.var(axis=0))[::-1]

    unique_states = sorted(np.unique(cell_states))
    if cell_fractions is not None and spot_fractions is not None:
        unique_states = [
            k
            for k in unique_states
            if cell_fractions.get(k, 0.0) > _FRAC_THRESHOLD
            and spot_fractions.get(k, 0.0) > _FRAC_THRESHOLD
        ]
    if not unique_states:
        logger.warning(
            "No states above fraction threshold — skipping state-profile plot."
        )
        return
    K = len(unique_states)

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
        figsize=(max(12, n_genes * 0.18) + 3 * (n_panels - 1), max(3, K * 0.5 + 2)),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    ax_heat = axes[0]
    im = ax_heat.imshow(mat_z, aspect="auto", cmap="viridis")
    ax_heat.set_xticks(range(n_genes))
    ax_heat.set_xticklabels(gene_names[gene_order], rotation=90, fontsize=5)
    ax_heat.set_yticks(range(K))
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
        ax.barh(range(K), values, color=bar_colors, alpha=0.8)
        for i, v in enumerate(values):
            ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=7)
        ax.set_yticks(range(K))
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

    if cell_fractions and spot_fractions:
        unique_states = [
            k
            for k in unique_states
            if cell_fractions.get(k, 0.0) > _FRAC_THRESHOLD
            and spot_fractions.get(k, 0.0) > _FRAC_THRESHOLD
        ]
    if not unique_states:
        logger.warning("No states above fraction threshold — skipping fractions plot.")
        return

    K = len(unique_states)
    fig, axes = plt.subplots(
        1,
        len(active),
        figsize=(3.5 * len(active), max(3, K * 0.45 + 1.5)),
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
        ax.barh(range(K), values, color=bar_colors, alpha=0.8)
        for i, v in enumerate(values):
            ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=8)
        ax.set_yticks(range(K))
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
    cell_fractions: dict[int, float] | None = None,
    spot_fractions: dict[int, float] | None = None,
) -> None:
    """
    Scatter plot of ST spots in physical space, coloured by computed cell-state.

    Coordinates are read from adata_st.obsm["spatial"].
    Only states with cell- and spot-fraction > _FRAC_THRESHOLD are shown when
    fractions are supplied.  state_palette syncs colours with the UMAP.
    """
    if "spatial" not in adata_st.obsm:
        logger.warning("adata_st has no obsm['spatial'] — skipping spatial plot.")
        return

    coords = np.asarray(adata_st.obsm["spatial"])
    unique_states = sorted(np.unique(spot_states).tolist())

    # Filter to states above the fraction threshold
    if cell_fractions is not None and spot_fractions is not None:
        unique_states = [
            s
            for s in unique_states
            if cell_fractions.get(s, 0.0) > _FRAC_THRESHOLD
            and spot_fractions.get(s, 0.0) > _FRAC_THRESHOLD
        ]
    if not unique_states:
        logger.warning("No states above fraction threshold — skipping spatial plot.")
        return

    K = len(unique_states)

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
        f"Spatial distribution of computed cell states  ({K} > 1 %)", fontsize=12
    )

    # Legend: compact when many states
    legend_fs = max(5, min(9, 120 // K))
    ncol = max(1, K // 20)
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


def plot_greedy_matched_fingerprints(
    adata_sc: AnnData,
    cell_states: np.ndarray,
    leiden_labels: np.ndarray,
    shared_genes: list[str],
    match_results: dict,
    output_dir: Path,
    output_subdir: str = "greedy_matched_fingerprints",
) -> None:
    """
    For every computed cell state, plot its mean-expression fingerprint alongside
    the greedy-matched Leiden cluster fingerprint.

    Layout per panel: two rows sharing the same gene axis —
        row 0  computed state k   (mean expression, z-scored per row)
        row 1  matched Leiden cluster  (same z-scoring)

    Outputs
    -------
    output_dir/greedy_matched_fingerprints/
        match_S<k>_L<l>.png   — one PNG per computed state
        overview.png           — all matches stacked vertically
    """
    available = [g for g in shared_genes if g in adata_sc.var_names]
    if len(available) < 2:
        logger.warning(
            "Too few shared genes for greedy fingerprint comparison — skipping."
        )
        return

    X = _dense_X(adata_sc[:, available])
    gene_names = np.array(available)
    gene_order = np.argsort(X.var(axis=0))[::-1]
    n_genes = len(available)

    best_leiden_per_computed = match_results["best_leiden_per_computed"]
    sim = match_results["sim_matrix"]
    n_computed = match_results["n_computed"]

    match_dir = output_dir / output_subdir
    match_dir.mkdir(parents=True, exist_ok=True)

    def _zrow(v: np.ndarray) -> np.ndarray:
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v - v.mean()

    all_panels: list[tuple[int, int, np.ndarray]] = []

    for state_k in range(n_computed):
        leiden_l = int(best_leiden_per_computed[state_k])
        cos_sim = float(sim[leiden_l, state_k])

        computed_mean = X[cell_states == state_k][:, gene_order].mean(axis=0)
        leiden_mean = X[leiden_labels == leiden_l][:, gene_order].mean(axis=0)
        mat = np.stack([_zrow(computed_mean), _zrow(leiden_mean)])
        all_panels.append((state_k, leiden_l, mat))

        fig, ax = plt.subplots(figsize=(max(12, n_genes * 0.18), 3))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-3, vmax=3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"Computed S{state_k}", f"Leiden L{leiden_l}"], fontsize=9)
        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(gene_names[gene_order], rotation=90, fontsize=5)
        ax.set_xlabel("Gene  (sorted by SC variance, z-scored per row)", fontsize=9)
        ax.set_title(
            f"Greedy match: Computed State {state_k}  ↔  Leiden {leiden_l}"
            f"   (cosine sim = {cos_sim:.3f})",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, label="z-score", fraction=0.02, pad=0.01)
        fig.tight_layout()
        fig.savefig(
            match_dir / f"match_S{state_k}_L{leiden_l}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    row_h = 2.2
    n_panels = len(all_panels)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(max(12, n_genes * 0.18), row_h * n_panels),
        gridspec_kw={"hspace": 0.8},
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (state_k, leiden_l, mat) in zip(axes, all_panels):
        cos_sim = float(sim[leiden_l, state_k])
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-3, vmax=3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"S{state_k}", f"L{leiden_l}"], fontsize=7)
        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(gene_names[gene_order], rotation=90, fontsize=5)
        ax.set_title(
            f"S{state_k} ↔ L{leiden_l}  (cos={cos_sim:.3f})",
            fontsize=8,
            pad=3,
        )
        fig.colorbar(im, ax=ax, label="z", fraction=0.01, pad=0.005)

    fig.suptitle(
        "Greedy-matched fingerprints — Computed states vs Leiden clusters",
        fontsize=11,
        y=1.001,
    )
    fig.savefig(match_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Greedy matched fingerprints → %s", match_dir)
