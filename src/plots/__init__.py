"""Plotting for the AIM pipeline — all matplotlib figure generation lives here.

Pure rendering: takes already-computed numbers / AnnData objects (from
``metrics`` and the ``analysis`` orchestration) and writes PNGs to disk. Kept
separate from ``metrics`` (which does computation only) and ``analysis`` (which
orchestrates) so both the AIM post-mapping analysis and the reference-aligner
analysis draw the shared figures from one place.

    onehot  plot_onehot_distribution, plot_dominance_thresholds
    cossim  plot_cossim_boxplots
    states  UMAP comparison/grid, Leiden merge map, state profiles/fractions,
            spatial cell states
"""

from .cossim import plot_cossim_boxplots
from .onehot import plot_dominance_thresholds, plot_onehot_distribution
from .states import (
    _build_state_palette,
    plot_leiden_merge_map,
    plot_spatial_cell_states,
    plot_state_fractions,
    plot_state_profiles,
    plot_umap_comparison,
    plot_umap_grid,
)

__all__ = [
    "plot_onehot_distribution",
    "plot_dominance_thresholds",
    "plot_cossim_boxplots",
    "_build_state_palette",
    "plot_umap_comparison",
    "plot_umap_grid",
    "plot_leiden_merge_map",
    "plot_state_profiles",
    "plot_state_fractions",
    "plot_spatial_cell_states",
]
