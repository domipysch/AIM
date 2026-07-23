from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from anndata import AnnData

from adata_schema import OBS_COMPUTED_STATE
from .biology import (
    analyse_substate_coherence,
    analyse_modularities,
    plot_modularities,
)
from .reconstruction import analyse_reconstruction, plot_reconstruction
from .states import create_states_plots
from .topology import analyse_spatial_organization

from .loading import (
    load_spot_to_state_mapping_soft_and_hard,
    load_leiden_to_state,
    infer_cell_to_state_cluster,
)
from .onehot import analyse_spot_to_state_one_hotness, plot_spot_to_state_one_hotness
from plots import (
    _build_state_palette,
    plot_spatial_cell_states,
    plot_nhood_enrichment,
)
from .report import generate_analysis_report

logger = logging.getLogger(__name__)


def run_analysis(
    adata_sc: AnnData, adata_st: AnnData, run_dir: Path, generate_pdf: bool
) -> None:
    """Run the full post-mapping analysis for one K and write the PDF report.

    Loads this K's mapping, computes state stats, one-hotness, spatial
    organisation, substate coherence, reconstruction cosine similarity and
    modularity, renders the plots, and compiles ``analysis/report.pdf`` inside
    ``run_dir``.

    Requires: adata_sc.obs[OBS_LEIDEN_ALL_GENES],
        adata_sc.obs[OBS_LEIDEN_SHARED_GENES],
        adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM] (and the further
        uns keys read by the delegate analyses). run_dir must contain
        spot_to_state_mapping_soft.h5ad and leiden_to_state.csv.
    Adds: adata_st.obsm[OBSM_MAPPING_SOFT], adata_st.obs[OBS_MAPPING_HARD],
        adata_sc.obs[OBS_COMPUTED_STATE].
    """

    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    data_dir = analysis_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load leiden to state mapping & infer cell to state mapping
    leiden_to_state = load_leiden_to_state(run_dir)
    infer_cell_to_state_cluster(adata_sc, leiden_to_state)

    # Load spot to state mapping
    load_spot_to_state_mapping_soft_and_hard(run_dir, adata_st)

    logger.info("Computing one-hot metrics...")
    analyse_spot_to_state_one_hotness(adata_st, data_dir)

    logger.info("Computing spatial organisation of mapped spots...")
    analyse_spatial_organization(adata_st, data_dir)

    logger.info("Computing substate merge coherence...")
    analyse_substate_coherence(adata_sc, leiden_to_state, data_dir)

    logger.info("Computing reconstruction cosine similarities...")
    analyse_reconstruction(adata_sc, adata_st, leiden_to_state, data_dir)

    logger.info("Computing modularity for computed assignment...")
    analyse_modularities(adata_sc, adata_st, data_dir)

    # Create PDF report
    if generate_pdf:

        # Compute state color palette
        state_palette = _build_state_palette(adata_sc)

        # Create plots
        create_states_plots(adata_sc, adata_st, plots_dir, state_palette=state_palette)
        plot_spot_to_state_one_hotness(plots_dir, data_dir)
        plot_reconstruction(plots_dir, data_dir)
        plot_modularities(adata_sc, plots_dir, data_dir, state_palette=state_palette)
        plot_nhood_enrichment(adata_st, plots_dir)
        plot_spatial_cell_states(
            adata_st,
            output_dir=plots_dir,
            state_palette=state_palette,
        )

        generate_analysis_report(analysis_dir)
        logger.info("Analysis report written to %s", analysis_dir)
