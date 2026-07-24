"""Matplotlib figure generation for the AIM pipeline; renders precomputed values and AnnData to PNGs."""

from .confidence import plot_confidence_distribution
from .cossim import plot_cossim_boxplots
from .onehot import plot_dominance_thresholds, plot_onehot_distribution
from .states import (
    _build_state_palette,
    plot_leiden_merge_map,
    plot_nhood_enrichment,
    plot_spatial_cell_states,
    plot_state_fractions,
    plot_state_profiles,
    plot_umap_comparison,
    plot_umap_grid,
)

__all__ = [
    "plot_confidence_distribution",
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
    "plot_nhood_enrichment",
]
