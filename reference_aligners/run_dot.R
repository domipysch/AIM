#!/usr/bin/env Rscript

# Run DOT alignment on a prepared dataset.
# Reads sc.h5ad and st.h5ad; saves all four outputs (prob + det GEP and mapping) as CSVs.
# Python converts the CSVs to h5ad after this script exits.
#
# Usage:
# Rscript run_dot.R <sc_path> <st_path> <LSO|HSO> <cellTypeKey|cellID> \
#                   <gep_prob_path.csv> <gep_det_path.csv> \
#                   <mapping_prob_path.csv> <mapping_det_path.csv>
#
# cellTypeKey:
#   "cellID"        -> map individual cells (each cell is its own type)
#   any obs column  -> aggregate cells by that column before mapping

library(DOTr)
library(hdf5r)
library(Matrix)
library(stats)
library(utils)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop("Need 8 args: sc_path, st_path, mode (LSO|HSO), cellTypeKey, gep_prob_path, gep_det_path, mapping_prob_path, mapping_det_path.")
}
sc_path          <- args[1]
st_path          <- args[2]
mode             <- toupper(args[3])
cellTypeKey      <- args[4]
gep_prob_path    <- args[5]
gep_det_path     <- args[6]
mapping_prob_path <- args[7]
mapping_det_path  <- args[8]

if (!mode %in% c("LSO", "HSO")) {
  stop(sprintf("Invalid mode '%s'. Use 'LSO' or 'HSO'.", mode))
}

# ---- Helper functions ----

# Read X from h5ad (handles dense and sparse CSR).
# Always returns a matrix of shape (n_obs x n_vars).
read_h5ad_X <- function(h5file) {
  x_obj <- h5file[["X"]]
  if (inherits(x_obj, "H5Group")) {
    # Sparse CSR: shape attribute holds [n_obs, n_vars] in Python/HDF5 order.
    # hdf5r does NOT transpose scalar/vector attributes, so shape[1]=n_obs, shape[2]=n_vars.
    data_vec <- x_obj[["data"]]$read()
    indices  <- x_obj[["indices"]]$read()
    indptr   <- x_obj[["indptr"]]$read()
    shape    <- h5attr(x_obj, "shape")
    n_obs  <- as.integer(shape[1])
    n_vars <- as.integer(shape[2])
    # CSR has n_obs+1 entries in indptr; CSC has n_vars+1.
    if (length(indptr) - 1L == n_obs) {
      m <- sparseMatrix(j = indices + 1L, p = indptr, x = as.numeric(data_vec),
                        dims = c(n_obs, n_vars), repr = "R")
    } else {
      m <- sparseMatrix(i = indices + 1L, p = indptr, x = as.numeric(data_vec),
                        dims = c(n_obs, n_vars), repr = "C")
    }
    return(as.matrix(m))  # n_obs x n_vars
  } else {
    # Dense: hdf5r reads HDF5 C-order (n_obs, n_vars) into R as (n_vars x n_obs) —
    # always transpose to recover (n_obs x n_vars).
    return(t(x_obj$read()))
  }
}

# Read the obs/var index (cell IDs / gene IDs / spot IDs).
read_h5ad_index <- function(group) {
  idx_name <- tryCatch(h5attr(group, "_index"), error = function(e) "_index")
  group[[idx_name]]$read()
}

# Read a single obs column; handles both plain string arrays and categorical encoding.
read_obs_col <- function(h5file, col_name) {
  obj <- h5file[[paste0("obs/", col_name)]]
  if (inherits(obj, "H5Group")) {
    # Categorical: codes (int, 0-indexed) + categories (string)
    cats  <- obj[["categories"]]$read()
    codes <- as.integer(obj[["codes"]]$read()) + 1L
    return(cats[codes])
  }
  obj$read()
}

# ---- Load sc.h5ad ----
cat("Loading sc.h5ad...\n")
sc_h5    <- H5File$new(sc_path, mode = "r")
cell_ids <- read_h5ad_index(sc_h5[["obs"]])  # length C
sc_gene_ids <- read_h5ad_index(sc_h5[["var"]])  # length G

X_sc <- read_h5ad_X(sc_h5)  # C x G
rownames(X_sc) <- cell_ids
colnames(X_sc) <- sc_gene_ids

# Cell type annotation:
#   cellTypeKey == "cellID" -> map individual cells, use obs index as annotation
#   otherwise               -> use the named obs column
if (cellTypeKey == "cellID") {
  cat("Mapping individual cells (cellTypeKey = 'cellID').\n")
  cell_types <- cell_ids
} else {
  available_cols <- names(sc_h5[["obs"]])
  if (!cellTypeKey %in% available_cols) {
    stop(sprintf(
      "cellTypeKey '%s' not found in sc.h5ad obs. Available columns: %s",
      cellTypeKey, paste(available_cols, collapse = ", ")
    ))
  }
  cat(sprintf("Mapping by cell type: '%s'.\n", cellTypeKey))
  cell_types <- read_obs_col(sc_h5, cellTypeKey)
}
sc_h5$close_all()

# ref_counts for DOT: G x C
ref_counts <- t(X_sc)
rownames(ref_counts) <- sc_gene_ids
colnames(ref_counts) <- cell_ids

# ---- Load st.h5ad ----
cat("Loading st.h5ad...\n")
st_h5        <- H5File$new(st_path, mode = "r")
spot_ids     <- read_h5ad_index(st_h5[["obs"]])  # length S
st_gene_ids  <- read_h5ad_index(st_h5[["var"]])  # length G (st's own gene list)
n_spots      <- length(spot_ids)

X_st <- read_h5ad_X(st_h5)  # S x G
rownames(X_st) <- spot_ids
colnames(X_st) <- st_gene_ids

# Spatial coordinates: stored as (S x 2); transpose if hdf5r returns (2 x S)
coords_raw <- st_h5[["obsm/spatial"]]$read()
if (nrow(coords_raw) == n_spots && ncol(coords_raw) == 2) {
  srt_coords <- coords_raw
} else {
  srt_coords <- t(coords_raw)
}
rownames(srt_coords) <- spot_ids
colnames(srt_coords) <- c("cArray0", "cArray1")
st_h5$close_all()

# srt_counts for DOT: G x S
srt_counts <- t(X_st)
rownames(srt_counts) <- st_gene_ids
colnames(srt_counts) <- spot_ids

# ---- Build reference centroids (T x G) ----
cat("Build reference centroids...\n")
celltype_levels    <- unique(cell_types)
ref_centroids_list <- lapply(celltype_levels, function(ct) {
  idx <- which(cell_types == ct)
  if (length(idx) == 1) {
    as.numeric(X_sc[idx, , drop = TRUE])
  } else {
    colMeans(X_sc[idx, , drop = FALSE])
  }
})
ref_centroids <- do.call(rbind, ref_centroids_list)  # T x G
rownames(ref_centroids) <- as.character(celltype_levels)
colnames(ref_centroids) <- sc_gene_ids

# ---- Create DOT object and run decomposition ----
cat("Create DOT object...\n")
dot.srt <- setup.srt(srt_data = srt_counts, srt_coords = srt_coords)
dot.ref <- setup.ref(ref_data = ref_counts, ref_annotations = cell_types, 1)
dot     <- create.DOT(dot.srt, dot.ref)

if (mode == "HSO") {
  cat("Running DOT in high-resolution mode (HSO)...\n")
  dot <- run.DOT.highresolution(dot)  # dot@weights: S x T
} else {
  cat("Running DOT in low-resolution mode (LSO)...\n")
  dot <- run.DOT.lowresolution(dot, ratios_weight = 0, max_spot_size = 20, verbose = FALSE)
  # dot@weights: S x T
}

# ---- Derive probabilistic and deterministic weights ----
weights_prob <- as.matrix(dot@weights)  # S x T

# Deterministic: argmax one-hot per spot
finite_mask  <- is.finite(weights_prob)
weights_repl <- weights_prob
weights_repl[!finite_mask] <- -Inf
argmax_idx   <- max.col(weights_repl, ties.method = "first")
weights_det  <- matrix(0, nrow(weights_prob), ncol(weights_prob),
                       dimnames = dimnames(weights_prob))
weights_det[cbind(seq_len(nrow(weights_prob)), argmax_idx)] <- 1
rows_no_finite <- rowSums(finite_mask) == 0
if (any(rows_no_finite)) weights_det[rows_no_finite, ] <- 0

# ---- Helper: compute and write GEP CSV (G x S) ----
write_gep <- function(weights, path) {
  gep <- t(ref_centroids) %*% t(weights)  # G x S
  gep <- as.matrix(gep)
  rownames(gep) <- sc_gene_ids
  colnames(gep) <- spot_ids
  out_df <- data.frame(GEP = rownames(gep), gep, check.names = FALSE, stringsAsFactors = FALSE)
  numeric_cols <- vapply(out_df, is.numeric, logical(1))
  out_df[numeric_cols] <- lapply(out_df[numeric_cols], function(x) round(x, 4))
  write.csv(out_df, file = path, row.names = FALSE, quote = FALSE)
}

# ---- Write all four outputs ----
out_dir <- dirname(gep_prob_path)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_gep(weights_prob, gep_prob_path)
cat("Done. GEP (prob) written to:", gep_prob_path, "\n")

write_gep(weights_det, gep_det_path)
cat("Done. GEP (det) written to:", gep_det_path, "\n")

write.csv(as.data.frame(weights_prob), file = mapping_prob_path, quote = FALSE)
cat("Done. Mapping (prob) written to:", mapping_prob_path, "\n")

write.csv(as.data.frame(weights_det), file = mapping_det_path, quote = FALSE)
cat("Done. Mapping (det) written to:", mapping_det_path, "\n")
