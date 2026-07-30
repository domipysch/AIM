from __future__ import annotations

import logging
from pathlib import Path

from anndata import AnnData

from .biology import (
    analyse_substate_coherence,
    analyse_modularities,
)
from .confidence import analyse_spot_confidence
from .reconstruction import analyse_reconstruction
from .topology import analyse_spatial_organization

from .loading import (
    load_spot_to_state_mapping_soft_and_hard,
    load_leiden_to_state,
    infer_cell_to_state_cluster,
)
from .onehot import analyse_spot_to_state_one_hotness

logger = logging.getLogger(__name__)


def run_analysis(adata_sc: AnnData, adata_st: AnnData, run_dir: Path) -> None:
    """Run the full post-mapping analysis for one K, writing machine-readable
    metrics under ``run_dir/analysis/data/``.

    Loads this K's mapping and computes state stats, one-hotness, spatial
    organisation, substate coherence, reconstruction cosine similarity and
    modularity. No figures are rendered: the interactive GUI (``python -m gui``)
    builds every plot on demand from these outputs.

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
    data_dir = analysis_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load leiden to state mapping & infer cell to state mapping
    leiden_to_state = load_leiden_to_state(run_dir)
    infer_cell_to_state_cluster(adata_sc, leiden_to_state)

    # Load spot to state mapping
    load_spot_to_state_mapping_soft_and_hard(run_dir, adata_st)

    logger.info("Computing one-hot metrics...")
    analyse_spot_to_state_one_hotness(adata_st, data_dir)

    logger.info("Summarising per-spot mapping confidence (if available)...")
    analyse_spot_confidence(adata_st, data_dir)

    logger.info("Computing spatial organisation of mapped spots...")
    analyse_spatial_organization(adata_st, data_dir)

    logger.info("Computing substate merge coherence...")
    analyse_substate_coherence(adata_sc, leiden_to_state, data_dir)

    logger.info("Computing reconstruction cosine similarities...")
    analyse_reconstruction(adata_sc, adata_st, leiden_to_state, data_dir)

    logger.info("Computing modularity for computed assignment...")
    analyse_modularities(adata_sc, adata_st, data_dir)
