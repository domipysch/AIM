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
    """
    Score one K's saved mapping outputs against ``adata_sc``/``adata_st``:
    one-hotness metrics, hard (argmax) mapping, reconstruction cosine
    similarity, biology metrics (spatial organisation + substate merge
    coherence), hard assignments/fractions/modularity of the computed-state
    partition, and the UMAP/spatial/state-profile plots — then writes the
    PDF report.

    Args:
        adata_sc: Single-cell AnnData (cells x genes), raw counts in .X;
                  carries obs[OBS_LEIDEN_ALL_GENES / OBS_LEIDEN_SHARED_GENES],
                  uns[UNS_SHARED_GENES / UNS_LEIDEN_RESOLUTION_ALL_GENES /
                  UNS_LEIDEN_SIZES / UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM /
                  UNS_LEIDEN_EXPR_SUMS_SHARED_GENES*] (see adata_schema.py) —
                  written by aim.clustering / aim.aggregation before the K
                  sweep runs, and read back here rather than recomputed. The
                  Leiden overclustering used to train this run is read from
                  here, not recomputed: scanpy's Leiden isn't deterministic
                  run-to-run, so a fresh clustering wouldn't reproduce the
                  actual partition the model was trained on.
        adata_st: Spatial AnnData (spots x genes), carrying layers[LAYER_LOGNORM].
                  Mutated in place: this K's mapping is loaded onto
                  adata_st.obsm[OBSM_MAPPING_SOFT] and obs[OBS_MAPPING_HARD].
        run_dir: one K_<kkk> sweep folder, containing spot_to_state_mapping.h5ad
                  and leiden_to_state.csv (as written by main.py). analysis/ is
                  written inside this folder.
    """

    # Get base paths
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    data_dir = analysis_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load this K's prob spot to state mapping (onto adata_st.obsm) and tree cut (leiden to state)
    load_mapping(run_dir, adata_st)
    leiden_to_state = load_leiden_to_state(run_dir)  # (L,)

    logger.info("Computing hard (argmax) assignments...")
    adata_st.obs[OBS_MAPPING_HARD] = adata_st.obsm[OBSM_MAPPING_SOFT].argmax(axis=1)
    leiden_idx = (
        adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    )  # (n_cells,) cell -> leiden
    cell_states = leiden_to_state[leiden_idx]  # (n_cells,) cell -> state
    adata_sc.obs[OBS_COMPUTED_STATE] = pd.Categorical(cell_states.astype(str))
    state_palette = _build_state_palette(sorted(np.unique(cell_states).tolist()))

    # ── 1. Cell state and spot state stats ──────────────────────
    create_states_plots(adata_sc, adata_st, plots_dir, state_palette=state_palette)

    # ── 2. One-hotness analysis ──────────────────────────────
    logger.info("Computing one-hot metrics...")
    analyse_spot_to_state_one_hotness(adata_st, plots_dir, data_dir)

    # ── 3. Topology analysis ────────────────────────
    logger.info("Computing spatial organisation of mapped spots...")
    analyse_spatial_organization(
        adata_st, data_dir, plots_dir, state_palette=state_palette
    )

    # ── 4. Biology analysis ────────────────────────
    logger.info("Computing substate merge coherence...")
    analyse_substate_coherence(
        adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM], leiden_to_state, data_dir
    )

    # ── 5. Reconstruction cosine similarity (soft/hard P x raw/norm) ────────
    logger.info("Computing reconstruction cosine similarities...")
    analyse_reconstruction(adata_sc, adata_st, leiden_to_state, data_dir, plots_dir)

    # ── 6. Modularity for the computed assignment ──────────────────────────────
    logger.info("Computing modularity for computed assignment...")
    analyse_modularities(adata_sc, data_dir, plots_dir, state_palette=state_palette)

    # Create PDF report
    generate_analysis_report(analysis_dir)
    logger.info("Analysis report written to %s", analysis_dir)
