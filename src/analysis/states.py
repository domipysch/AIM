from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from anndata import AnnData

from adata_schema import (
    OBS_MAPPING_HARD,
    OBS_COMPUTED_STATE,
    OBS_LEIDEN_ALL_GENES,
    OBSM_MAPPING_SOFT,
)

from .utils import cell_state_fractions
from plots import (
    plot_leiden_merge_map,
    plot_state_profiles,
    plot_state_fractions,
)

logger = logging.getLogger(__name__)


def create_states_plots(
    adata_sc: AnnData,
    adata_st: AnnData,
    output_plots_dir: Path,
    state_palette: dict[int, tuple] | None = None,
):
    """Plot cell-state profiles, state fractions, and the Leiden->state merge map.

    Requires: adata_st.obsm[OBSM_MAPPING_SOFT], adata_st.obs[OBS_MAPPING_HARD],
        adata_sc.obs[OBS_LEIDEN_ALL_GENES], adata_sc.obs[OBS_COMPUTED_STATE].
    Writes cell_state_profiles.png, cell_state_fractions.png and
    leiden_merge_map.png under output_plots_dir.
    """

    k = adata_st.obsm[OBSM_MAPPING_SOFT].shape[1]
    leiden_idx = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()
    cell_fractions = cell_state_fractions(cell_states, k)
    spot_fractions = cell_state_fractions(adata_st.obs[OBS_MAPPING_HARD].to_numpy(), k)

    logger.info("Plotting cell-state profiles...")
    plot_state_profiles(
        adata_sc,
        cell_states,
        output_plots_dir / "cell_state_profiles.png",
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
        state_palette=state_palette,
    )
    plot_state_fractions(
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
        unique_states=sorted(np.unique(cell_states).tolist()),
        output_path=output_plots_dir / "cell_state_fractions.png",
        state_palette=state_palette,
    )

    plot_leiden_merge_map(
        leiden_idx,
        cell_states,
        output_plots_dir / "leiden_merge_map.png",
        state_palette=state_palette,
    )
