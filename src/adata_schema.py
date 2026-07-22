"""
Single source of truth — and a map — for every obs/var/uns/obsm/obsp/layers
key the AIM sweep and its post-mapping analysis read or write on the sc/st
AnnData objects. Two purposes in one file:

  1. Avoid duplicating the same string key in the module that writes it and
     the module(s) that read it back (e.g. aim.aggregation writes
     UNS_LEIDEN_SIZES on adata_sc.uns; analysis.run_from_output reads it back
     — both import the same name from here instead of retyping the string).
  2. Read top to bottom, this file is a schema: everything that ends up on
     adata_sc / adata_st — and on the one read-only working copy the
     post-mapping analysis derives from them — by the time a sweep has run.

Has no dependencies of its own, so both the ``aim`` and ``analysis`` packages
can import from it freely without a circular import — ``aim.sweep`` already
imports ``analysis.run_from_output``, so the reverse (``analysis`` importing
directly from ``aim``) would deadlock on that cycle.
"""

# ── adata_sc — mutated in place at every stage, never copied ───────────────

# obs — written by aim.clustering.run_leiden_clustering.
OBS_LEIDEN_ALL_GENES = (
    "leiden"  # (n_cells,) int — the Leiden over-clustering label (0..L-1)
)
OBS_LEIDEN_SHARED_GENES = (
    "leiden_shared_genes"  # (n_cells,) int — the Leiden over-clustering label (0..L-1)
)

# uns — written by aim.clustering.run_leiden_clustering, alongside obs[OBS_LEIDEN].
# Read back by analysis.run_from_output.prepare_experiment instead of
# re-reading the run's config.yaml (main.py still writes leiden_resolution
# there too, but only as a human-readable provenance record — not read back).
UNS_LEIDEN_RESOLUTION_ALL_GENES = (
    "leiden_resolution"  # float — resolution the Leiden over-clustering was run at
)
UNS_LEIDEN_NUMBER_STATES_ALL_GENES = "leiden_number_states"

# uns — written by aim.clustering.run_leiden_clustering_shared_genes, alongside
# obs[OBS_LEIDEN_SHARED_GENES]. Mirrors UNS_LEIDEN_RESOLUTION/NUMBER_STATES
# above but for the shared-genes-only clustering run.
UNS_LEIDEN_RESOLUTION_SHARED_GENES = "leiden_resolution_shared_genes"  # float — resolution the shared-genes Leiden over-clustering was run at
UNS_LEIDEN_NUMBER_STATES_SHARED_GENES = "leiden_number_states_shared_genes"

# obsm — PCA embedding of the shared-genes-only normalized matrix
# (obsm[OBSM_LOGNORM_SHARED_GENES]), written by
# aim.clustering.run_leiden_clustering_shared_genes. Mirrors OBSM_PCA below,
# which is computed on all genes instead.
OBSM_PCA_SHARED_GENES = "X_pca_shared_genes"

# uns/obsp — neighbor graph over obsm[OBSM_PCA_SHARED_GENES], written by the
# same function. uns[UNS_NEIGHBORS_SHARED_GENES] holds scanpy's neighbors
# params dict; obsp[OBSP_DISTANCES_SHARED_GENES] /
# obsp[OBSP_CONNECTIVITIES_SHARED_GENES] hold the actual graph matrices
# (scanpy's own key_added + "_distances" / "_connectivities" naming
# convention — see sc.pp.neighbors(key_added=...)).
UNS_NEIGHBORS_SHARED_GENES = "neighbors_shared_genes"
OBSP_DISTANCES_SHARED_GENES = "neighbors_shared_genes_distances"
OBSP_CONNECTIVITIES_SHARED_GENES = "neighbors_shared_genes_connectivities"

# layers — written by aim.clustering.run_leiden_clustering on adata_sc (X stays
# the raw counts main.py loaded); also written on adata_st by
# analysis.run_from_output.prepare_experiment (see the adata_st section below)
# — same key, same transform, on whichever object needs it.
LAYER_LOGNORM = (
    "lognorm"  # (n_obs x n_genes) — normalize_total(target_sum=1e4) + log1p, all genes
)

# obsm — normalize_total(target_sum=1e4) + log1p computed on the shared genes
# only (sc/st gene intersection), so the size factor differs from
# LAYER_LOGNORM (which normalizes over all genes first, then gets sliced down
# elsewhere). Written on both adata_sc and adata_st by aim.sweep.pre_processing.
# Column count is G_shared, not n_vars, so this can't live in .layers; column
# order matches UNS_SHARED_GENES (defined below).
OBSM_LOGNORM_SHARED_GENES = "lognorm_shared"

# obsm — scanpy's own convention name, but referenced explicitly by name in
# both aim.clustering (use_rep=) and analysis.plots (plot_umap_grid's basis),
# so it's named here rather than left as a bare string in both.
OBSM_PCA = "X_pca"
OBSM_UMAP = "X_umap"
OBSM_UMAP_SHARED_GENES = "X_umap_shared_genes"

# uns — raw and normalized variants alike are written once, by
# aim.aggregation.compute_leiden_aggregates, before the K sweep runs. All
# four arrays are column-aligned to UNS_SHARED_GENES.
UNS_SHARED_GENES = "shared_genes"  # list[str] — sc/st gene intersection; the exact
# column order every array below is aligned to.
UNS_LEIDEN_EXPR_SUMS_SHARED_GENES = "leiden_expr_sums_shared"  # (L x G_shared) — summed raw expression per Leiden cluster
UNS_LEIDEN_SIZES = "leiden_sizes"  # (L,) — number of cells per Leiden cluster
UNS_LEIDEN_CENTROIDS_SHARED_GENES = (
    "leiden_centroids_shared"  # (L x G_shared) — per-cluster mean raw expression
)
UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM = "leiden_expr_sums_shared_norm"  # (L x G_shared) — summed normalized expression per cluster
UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM = "leiden_centroids_shared_norm"  # (L x G_shared) — per-cluster mean normalized expression

# ── adata_st — read-only from main.py's h5ad load through the whole sweep,
# except for one addition: ─────────────────────────────────────────────────

# obsm — supplied by the input h5ad itself (see the ST h5ad format
# requirements in CLAUDE.md); read by biology_metrics/plots for the spatial plot.
OBSM_SPATIAL = (
    "spatial"  # (n_spots x 2) — spatial coordinates, as loaded from the input h5ad
)

# obsm — one K's spot -> computed-state soft mapping (P, S x k), written by
# analysis.mapping_metrics.load_mapping when it reads spot_to_state_mapping.h5ad
# (as written by aim.io.write_run_outputs) back in for the post-mapping
# analysis. Overwritten once per K inside analysis.run_from_output.analyze_experiment's
# sweep loop — not meant to persist across K's. Column order is
# state_0..state_{k-1}; no separate name list is kept since nothing consumes
# state identity beyond that.
OBSM_MAPPING_SOFT = "mapping_soft"

# obsm — one K's spot -> winning-state index (S x 1, int, values 0..k-1) —
# i.e. argmax(OBSM_MAPPING_SOFT, axis=1), NOT one-hot encoded (obsm entries
# must be matrices, hence the (S, 1) shape rather than a plain (S,) array).
# Written by analysis.analysis.run_analysis alongside OBSM_MAPPING_SOFT;
# same overwrite-per-K lifetime.
OBSM_MAPPING_HARD = "mapping_hard"
