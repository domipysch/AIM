#!/usr/bin/env Rscript

# Run DOT alignment on a prepared dataset (always in high-resolution mode, HSO).
# Reads sc.h5ad and st.h5ad; saves the probabilistic mapping as a CSV.
# Python converts the CSV to h5ad after this script exits.
#
# Usage:
# Rscript run_dot.R <sc_path> <st_path> <cellTypeKey|cellID> <mapping_prob_path.csv>
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
if (length(args) < 4) {
  stop("Need 4 args: sc_path, st_path, cellTypeKey, mapping_prob_path.")
}
sc_path           <- args[1]
st_path           <- args[2]
cellTypeKey       <- args[3]
mapping_prob_path <- args[4]

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
# The index is normally a plain string dataset, but anndata writes a *group*
# whenever the pandas index carried an extension dtype: nullable-string-array
# (values + mask) or categorical (categories + codes). An H5Group has no $read(),
# so reading one blindly fails with "attempt to apply non-function".
read_h5ad_index <- function(group) {
  idx_name <- tryCatch(h5attr(group, "_index"), error = function(e) "_index")
  read_h5ad_str_array(group[[idx_name]])
}

# Resolve a string array that may be stored as a dataset or as an encoded group.
read_h5ad_str_array <- function(obj) {
  if (!inherits(obj, "H5Group")) {
    return(obj$read())
  }
  members <- names(obj)
  if ("values" %in% members) {          # nullable-string-array
    return(as.character(obj[["values"]]$read()))
  }
  if ("categories" %in% members && "codes" %in% members) {   # categorical
    cats  <- obj[["categories"]]$read()
    codes <- as.integer(obj[["codes"]]$read()) + 1L
    return(cats[codes])
  }
  stop(sprintf(
    "unsupported h5ad string encoding: group with members [%s]",
    paste(members, collapse = ", ")
  ))
}

# Read a single obs column; same three encodings as the index.
read_obs_col <- function(h5file, col_name) {
  read_h5ad_str_array(h5file[[paste0("obs/", col_name)]])
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

# ---- Create DOT object and run decomposition ----
cat("Create DOT object...\n")
dot.srt <- setup.srt(srt_data = srt_counts, srt_coords = srt_coords)
dot.ref <- setup.ref(ref_data = ref_counts, ref_annotations = cell_types, 1)
dot     <- create.DOT(dot.srt, dot.ref)

cat("Running DOT in high-resolution mode (HSO)...\n")
dot <- run.DOT.highresolution(dot)  # dot@weights: S x T

weights_prob <- as.matrix(dot@weights)  # S x T

# ---- Write mapping CSV ----
out_dir <- dirname(mapping_prob_path)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(as.data.frame(weights_prob), file = mapping_prob_path, quote = FALSE)
cat("Done. Mapping (prob) written to:", mapping_prob_path, "\n")
