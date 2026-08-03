"""Canonical obs/var/uns/obsm/obsp/layers key names for the sc/st AnnData
objects, each with a one-line description of what it holds. ``L`` = number of
Leiden subclusters, ``G_shared`` = number of shared genes, ``K`` = number of
states at the current cut.
"""

# ── adata_sc ───────────────────────────────────────────────────────────────

# obs
OBS_LEIDEN_ALL_GENES = "leiden"  # (n_cells,) int — Leiden label (0..L-1) from all genes
OBS_LEIDEN_SHARED_GENES = (
    "leiden_shared_genes"  # (n_cells,) int — Leiden label from shared genes only
)
OBS_COMPUTED_STATE = (
    "computed_state"  # (n_cells,) str categorical — cell's state at the current K
)

# uns — Leiden run parameters
UNS_LEIDEN_RESOLUTION_ALL_GENES = "leiden_resolution"  # float — resolution (all genes)
UNS_LEIDEN_NUMBER_STATES_ALL_GENES = (
    "leiden_number_states"  # int — cluster count L (all genes)
)
UNS_LEIDEN_RESOLUTION_SHARED_GENES = (
    "leiden_resolution_shared_genes"  # float — resolution (shared genes)
)
UNS_LEIDEN_NUMBER_STATES_SHARED_GENES = (
    "leiden_number_states_shared_genes"  # int — cluster count (shared genes)
)
UNS_MODULARITY_SHARED_LEIDEN = "modularity_shared_leiden"  # float — modularity of the shared-gene Leiden partition (K-independent; cached across the sweep)

# obsm/uns/obsp — shared-gene PCA + neighbor graph
OBSM_PCA_SHARED_GENES = (
    "X_pca_shared_genes"  # (n_cells x n_comps) — PCA of the shared-gene lognorm matrix
)
UNS_NEIGHBORS_SHARED_GENES = (
    "neighbors_shared_genes"  # dict — scanpy neighbors params for the shared-gene graph
)
OBSP_DISTANCES_SHARED_GENES = "neighbors_shared_genes_distances"  # (n_cells x n_cells) sparse — neighbor distances
OBSP_CONNECTIVITIES_SHARED_GENES = "neighbors_shared_genes_connectivities"  # (n_cells x n_cells) sparse — neighbor connectivities

# uns — per-Leiden-cluster aggregates, all column-aligned to UNS_SHARED_GENES
UNS_SHARED_GENES = "shared_genes"  # list[str] — sc/st gene intersection; column order for every *_SHARED array
UNS_LEIDEN_SIZES = "leiden_sizes"  # (L,) — cells per Leiden cluster
UNS_LEIDEN_EXPR_SUMS_SHARED_GENES = (
    "leiden_expr_sums_shared"  # (L x G_shared) — summed raw expression per cluster
)
UNS_LEIDEN_CENTROIDS_SHARED_GENES = "leiden_centroids_shared"  # (L x G_shared) — mean raw expression per cluster (all-zero rows set to 1e-6)
UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM = "leiden_expr_sums_shared_norm"  # (L x G_shared) — summed lognorm expression per cluster
UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM = "leiden_centroids_shared_norm"  # (L x G_shared) — mean lognorm expression per cluster

# ── adata_st ───────────────────────────────────────────────────────────────

# obsm/obs — one K's spot->state mapping
OBSM_SPATIAL = "spatial"  # (n_spots x 2) — spatial coordinates
OBSM_MAPPING_SOFT = (
    "mapping_soft"  # (n_spots x K) — soft spot->state assignment P at the current K
)
OBS_MAPPING_HARD = (
    "mapping_hard"  # (n_spots,) int — argmax state index (0..K-1) at the current K
)
OBS_MAPPING_STATE_CAT = "mapping_state"  # (n_spots,) str categorical — hard state label as a category, for squidpy neighbourhood tools (nhood_enrichment)
OBS_MAPPING_CONFIDENCE = "mapping_confidence"  # (n_spots,) float in [0,1] — per-spot assignment confidence at the current K; absent when the mapper defines no confidence (reference)

# obsm/obsp/uns — K-independent spatial/expression graphs, built once and reused across the K-sweep
OBSP_SPATIAL_CONNECTIVITIES = "spatial_connectivities"  # (n_spots x n_spots) sparse — squidpy spatial KNN graph (local purity + neighbourhood enrichment)
OBSP_ST_EXPR_CONNECTIVITIES = "expr_connectivities"  # (n_spots x n_spots) sparse — ST all-gene expression KNN graph for the mapping modularity
UNS_NHOOD_ENRICHMENT = f"{OBS_MAPPING_STATE_CAT}_nhood_enrichment"  # dict {zscore, count} — squidpy neighbourhood-enrichment result over the mapped states

# ── shared across adata_sc and adata_st ─────────────────────────────────────

LAYER_LOGNORM = (
    "lognorm"  # (n_obs x n_vars) — normalize_total(1e4) + log1p over all genes
)
OBSM_LOGNORM_SHARED_GENES = "lognorm_shared"  # (n_obs x G_shared) — normalize_total(1e4) + log1p over shared genes only, aligned to UNS_SHARED_GENES

# ── scanpy default keys, referenced by name ─────────────────────────────────

OBSM_PCA = "X_pca"  # (n_cells x n_comps) — PCA of the all-gene lognorm layer
OBSM_UMAP = "X_umap"  # (n_cells x 2) — UMAP from the all-gene neighbor graph
OBSM_UMAP_SHARED_GENES = (
    "X_umap_shared_genes"  # (n_cells x 2) — UMAP from the shared-gene neighbor graph
)
