"""Figure production for the GUI.

Two kinds of figures:

* The **spatial** plot is rendered live here (a small bespoke scatter) because the
  confidence-threshold slider must recolour spots on every change -- spots below
  the threshold are drawn grey.
* Every **report section** figure is produced by the repository's *existing* plot
  functions, called with data read straight off disk. Disk-only sections
  (one-hotness / reconstruction / confidence) need just the K's ``analysis/data``
  folder; scaffold sections (UMAPs / profiles / fractions / merge map) additionally
  need the reference scaffold. Rendered PNGs are cached under
  ``<output_dir>/.gui_cache/<mapper>/k_<kkk>/`` so re-viewing a K is instant.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are streamed, never shown in a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from anndata import AnnData  # noqa: E402

from analysis.biology import plot_modularities  # noqa: E402
from analysis.confidence import plot_spot_confidence  # noqa: E402
from analysis.loading import (  # noqa: E402
    infer_cell_to_state_cluster,
    load_leiden_to_state,
    load_spot_to_state_mapping_soft_and_hard,
)
from analysis.onehot import plot_spot_to_state_one_hotness  # noqa: E402
from analysis.reconstruction import plot_reconstruction  # noqa: E402
from analysis.states import create_states_plots  # noqa: E402
from plots import _build_state_palette  # noqa: E402

from . import data_access  # noqa: E402

logger = logging.getLogger(__name__)

_GREY = (0.72, 0.72, 0.72, 0.9)

# Logical name -> filename produced by the reused plot functions.
_DISK_PLOTS = {
    "onehot_distribution": "onehot_distribution_mapping.png",
    "onehot_thresholds": "onehot_thresholds_mapping.png",
    "reconstruction": "cossim_boxplots.png",
    "confidence": "confidence_distribution.png",
}
_SCAFFOLD_PLOTS = {
    "umap_computed_state": "umap_computed_state.png",
    "umap_grid": "umap_grid.png",
    "profiles": "cell_state_profiles.png",
    "fractions": "cell_state_fractions.png",
    "leiden_merge": "leiden_merge_map.png",
}


def state_palette(k: int) -> dict[int, tuple]:
    """tab20 palette keyed by state id, matching ``plots._build_state_palette``."""
    cmap = plt.get_cmap("tab20")
    return {s: cmap(s % 20) for s in range(k)}


def cache_plot_dir(output_dir: Path, mapper: str, k: int) -> Path:
    d = Path(output_dir) / ".gui_cache" / mapper / f"k_{k:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_spatial_fig(
    coords: np.ndarray,
    hard: np.ndarray,
    confidence: np.ndarray | None,
    threshold: float,
    k: int,
    dot_size: float = 8.0,
) -> "plt.Figure":
    """Scatter of ST spots coloured by hard state; spots with confidence below the
    threshold are drawn grey. Mirrors ``plots.plot_spatial_cell_states`` layout."""
    palette = state_palette(k)
    fig, ax = plt.subplots(figsize=(7, 6))

    if confidence is not None and threshold > 0.0:
        low = confidence < threshold
    else:
        low = np.zeros(len(hard), dtype=bool)

    n_low = int(low.sum())
    if n_low:
        ax.scatter(
            coords[low, 0],
            coords[low, 1],
            c=[_GREY],
            s=dot_size,
            linewidths=0,
            label=f"below threshold ({n_low})",
            rasterized=True,
        )

    keep = ~low
    for state in sorted(np.unique(hard[keep]).tolist()):
        mask = keep & (hard == state)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[palette.get(state, _GREY)],
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
    title = f"Spatial cell states (K={k})"
    if confidence is not None and threshold > 0.0:
        title += f"  |  confidence ≥ {threshold:.2f}"
    ax.set_title(title, fontsize=12)
    n_series = len(np.unique(hard[keep])) + (1 if n_low else 0)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=max(5, min(9, 120 // max(1, n_series))),
        ncol=max(1, n_series // 20),
        markerscale=1.5,
        frameon=False,
    )
    fig.tight_layout()
    return fig


def ensure_disk_plots(
    output_dir: Path, mapper: str, root: Path, k: int
) -> dict[str, Path]:
    """Render (if not cached) the disk-only report figures for one K; return the
    logical-name -> path map of those that exist."""
    plots_dir = cache_plot_dir(output_dir, mapper, k)
    ddir = data_access.data_dir(root, k)

    if not all(
        (plots_dir / fn).exists() for fn in ("onehot_distribution_mapping.png",)
    ):
        plot_spot_to_state_one_hotness(plots_dir, ddir)
    if not (plots_dir / "cossim_boxplots.png").exists():
        plot_reconstruction(plots_dir, ddir)
    # No-op if the mapper wrote no confidence.
    if not (plots_dir / "confidence_distribution.png").exists():
        plot_spot_confidence(plots_dir, ddir)

    return {
        name: plots_dir / fn
        for name, fn in _DISK_PLOTS.items()
        if (plots_dir / fn).exists()
    }


def ensure_scaffold_plots(
    adata_sc: AnnData,
    adata_st: AnnData,
    output_dir: Path,
    mapper: str,
    root: Path,
    k: int,
) -> dict[str, Path]:
    """Render (if not cached) the scaffold-dependent report figures for one K.

    Sets this K's state cut on the scaffold and loads P onto the ST object, then
    calls the existing ``create_states_plots`` and ``plot_modularities`` (which
    emit the two UMAP figures)."""
    plots_dir = cache_plot_dir(output_dir, mapper, k)
    ddir = data_access.data_dir(root, k)

    if not all((plots_dir / fn).exists() for fn in _SCAFFOLD_PLOTS.values()):
        leiden_to_state = load_leiden_to_state(data_access.k_dir(root, k))
        infer_cell_to_state_cluster(adata_sc, leiden_to_state)
        load_spot_to_state_mapping_soft_and_hard(data_access.k_dir(root, k), adata_st)
        palette = _build_state_palette(adata_sc)

        create_states_plots(adata_sc, adata_st, plots_dir, state_palette=palette)
        plot_modularities(adata_sc, plots_dir, ddir, state_palette=palette)

    return {
        name: plots_dir / fn
        for name, fn in _SCAFFOLD_PLOTS.items()
        if (plots_dir / fn).exists()
    }
