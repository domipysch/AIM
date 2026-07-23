from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from adata_schema import (
    OBS_COMPUTED_STATE,
    OBS_LEIDEN_ALL_GENES,
    OBSM_MAPPING_SOFT,
    UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM,
    OBS_MAPPING_HARD,
)
from .biology import analyse_substate_coherence, analyse_modularities
from .reconstruction import analyse_reconstruction
from .states import create_states_plots
from .topology import analyse_spatial_organization

from .loading import load_mapping, load_leiden_to_state
from .onehot import analyse_spot_to_state_one_hotness
from plots import (
    _build_state_palette,
)
from .report import generate_analysis_report

logger = logging.getLogger(__name__)


def run_analysis(adata_sc: AnnData, adata_st: AnnData, run_dir: Path) -> None:
    """Run the full post-mapping analysis for one K and write the PDF report.

    Loads this K's mapping, computes state stats, one-hotness, spatial
    organisation, substate coherence, reconstruction cosine similarity and
    modularity, renders the plots, and compiles ``analysis/report.pdf`` inside
    ``run_dir``.

    Requires: adata_sc.obs[OBS_LEIDEN_ALL_GENES],
        adata_sc.obs[OBS_LEIDEN_SHARED_GENES],
        adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM] (and the further
        uns keys read by the delegate analyses). run_dir must contain
        spot_to_state_mapping.h5ad and leiden_to_state.csv.
    Adds: adata_st.obsm[OBSM_MAPPING_SOFT], adata_st.obs[OBS_MAPPING_HARD],
        adata_sc.obs[OBS_COMPUTED_STATE].
    """

    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    data_dir = analysis_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    load_mapping(run_dir, adata_st)
    leiden_to_state = load_leiden_to_state(run_dir)

    logger.info("Computing hard (argmax) assignments...")
    adata_st.obs[OBS_MAPPING_HARD] = adata_st.obsm[OBSM_MAPPING_SOFT].argmax(axis=1)
    leiden_idx = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    cell_states = leiden_to_state[leiden_idx]
    adata_sc.obs[OBS_COMPUTED_STATE] = pd.Categorical(cell_states.astype(str))
    state_palette = _build_state_palette(sorted(np.unique(cell_states).tolist()))

    create_states_plots(adata_sc, adata_st, plots_dir, state_palette=state_palette)

    logger.info("Computing one-hot metrics...")
    analyse_spot_to_state_one_hotness(adata_st, plots_dir, data_dir)

    logger.info("Computing spatial organisation of mapped spots...")
    analyse_spatial_organization(
        adata_st, data_dir, plots_dir, state_palette=state_palette
    )

    logger.info("Computing substate merge coherence...")
    analyse_substate_coherence(
        adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM], leiden_to_state, data_dir
    )

    logger.info("Computing reconstruction cosine similarities...")
    analyse_reconstruction(adata_sc, adata_st, leiden_to_state, data_dir, plots_dir)

    logger.info("Computing modularity for computed assignment...")
    analyse_modularities(adata_sc, data_dir, plots_dir, state_palette=state_palette)

    # Create PDF report
    generate_analysis_report(analysis_dir)
    logger.info("Analysis report written to %s", analysis_dir)
