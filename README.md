# Spatial Transcriptomics Alignment

Maps scRNA-seq reference data onto high-resolution spatial transcriptomics (ST) spots by learning a joint cell-to-cell-state and spot-to-state assignment.

## Repository Structure

```
main.py                          # Novel method — single-pair entry point
environment.yml                  # Conda env for the novel method
src/
├── model.py                     # AlternativeIdeaModel (learnable A, B matrices)
├── loss.py                      # Multi-term loss (rec_spot, rec_gene, entropy, …)
├── dataset.py                   # h5ad → tensor preparation
├── spatial_graph.py             # KNN / Delaunay / Radius graph builders
├── sc_embedding.py              # PCA / scVI cell embeddings
├── utils.py                     # Shared helpers
├── evaluate_k/                  # Post-mapping analysis (clustering, reports, plots)
└── metrics/                     # Evaluation metrics O2, O4
reference_aligners/              # Baseline method wrappers (Tangram, TACCO, DOT)
batch_processing/
├── run_pre_check_all_pairs.py   # Batch pre-alignment checks
├── run_reference_aligner_all_pairs.py  # Batch baseline aligners
└── grid_search/
    ├── grid_search.py           # Grid search — single pair
    ├── grid_search_config.yaml  # Grid search hyperparameter config
    └── run_grid_search_all_pairs.py    # Grid search — all pairs, multi-GPU
data_preparation/                # Dataset utilities (validate, convert, split, …)
pre_check/                       # Pre-alignment compatibility diagnostics
sample_dataset/                  # Minimal dataset mirroring the database layout (scRNA/, ST/, pairs.csv)
```

## Input dataset

To use this method, you need pairs of scRNA-seq data with according spatial transcriptomics data.

### h5ad format

Both data modalities are expected in `.h5ad` format (raw counts, though the method can also be applied to pre-processed data).

- **scRNA `<Name>.h5ad`:** `X` = raw counts (cells × genes, float32); `var_names` = uppercase gene symbols; `obs` includes at least one cell-type column.
- **ST `<Name>.h5ad`:** `X` = raw counts (spots × genes, float32); `var_names` = uppercase gene symbols; `obsm["spatial"]` = float array (n_spots × 2).

### Data preparation single run

If just applying this method (or a reference method) to such a single pair of scRNA and ST data,
just have those two files available.

### Data preparation batch run

If you want to apply this method (or a reference method) to a whole list of dataset pairs),
structure your set of pairs the following way.

```
sample_dataset/
├── scRNA/
│   ├── index.csv          # one row per scRNA dataset
│   └── <scName1>.h5ad
│   └── <scName2>.h5ad
│   └── ...
├── ST/
│   ├── index.csv          # one row per ST dataset
│   └── <stName1>.h5ad
│   └── <stName2>.h5ad
│   └── ...
└── pairs.csv              # links scRNA ↔ ST (PairID, scName, stName, …)
```

This setup enables reusing one sc-dataset for multiple different spatial datasets,
by just referencing the sc-dataset multiple times in `pairs.csv`.
Feel free to add other columns to any of the `.csv` files holding any other information you have on the datasets.

See `/sample_dataset` for a minimal reference setup.

## Environments

Each method requires its own conda environment.

| Method | Environment | Create |
|--------|-------------|--------|
| Novel method | `alternative_idea_env` | `conda env create -f environment.yml` |
| Tangram | `tangram_env` | `conda env create -f reference_aligners/environment_tangram.yml` |
| TACCO | `tacco_env` | `conda env create -f reference_aligners/environment_tacco.yml` |
| DOT | `dot_env` | `conda env create -f reference_aligners/environment_dot.yml` then `Rscript -e "remotes::install_github('saezlab/DOT')"` |

---

## Usage

> All commands below are run from the **repository root**.

### 1. Pre-alignment compatibility check (single pair)

Generate statistics on your given input sc- and st-dataset pair. Does not map them yet.
Computes statistics as cell, spot and gene counts, cell/spot library sizes, etc. as well as first naive metrics on compatability for mapping.
Writes a full PDF report (generated with `typst`) about your dataset pair to the output folder.

```bash
conda activate alternative_idea_env
python -m pre_check \
    --scdata        <path/to/sc.h5ad> \
    --stdata        <path/to/st.h5ad> \
    --output_folder <output/pre_check/pair_0> \
    [--leiden_resolution 0.5]
```

**Example with sample dataset:**
```bash
conda activate alternative_idea_env
python -m pre_check \
    --scdata        sample_dataset/scRNA/sample_sc.h5ad \
    --stdata        sample_dataset/ST/sample_st.h5ad \
    --output_folder sample_output/pre_check/sample
```

---

### 2. Pre-alignment compatibility check (all pairs)

Runs the pre-check for every row in `pairs.csv` in parallel.

```bash
conda activate alternative_idea_env
python -m batch_processing.run_pre_check_all_pairs \
    --pairs_csv <path/to/pairs.csv> \
    --sc_dir    <path/to/scRNA> \
    --st_dir    <path/to/ST> \
    --output_dir <output/pre_check> \
    [--workers 4] \
    [--leiden_resolution 0.5]
```

**Example with sample dataset:**
```bash
conda activate alternative_idea_env
python -m batch_processing.run_pre_check_all_pairs \
    --pairs_csv sample_dataset/pairs.csv \
    --sc_dir    sample_dataset/scRNA \
    --st_dir    sample_dataset/ST \
    --output_dir sample_output/pre_check/sample
```

---

### 3. Run a reference aligner (single pair)

All three aligners share the same interface.

```bash
# Tangram
conda activate tangram_env
python -m reference_aligners.run_tangram \
    --scdata        <path/to/sc.h5ad> \
    --stdata        <path/to/st.h5ad> \
    --output_folder <sample_output/tangram/pair_0> \
    --cell_type_key <obs_column_with_cell_types>

# TACCO
conda activate tacco_env
python -m reference_aligners.run_tacco \
    --scdata        <path/to/sc.h5ad> \
    --stdata        <path/to/st.h5ad> \
    --output_folder <sample_output/tacco/pair_0> \
    --cell_type_key <obs_column_with_cell_types>

# DOT
conda activate dot_env
python -m reference_aligners.run_dot \
    --scdata        <path/to/sc.h5ad> \
    --stdata        <path/to/st.h5ad> \
    --output_folder <sample_output/dot/pair_0> \
    --cell_type_key <obs_column_with_cell_types>
```

Each aligner writes `gep_prob.h5ad` and `gep_det.h5ad` to the output folder.

**Example with sample dataset** (`cellType` and `cellTypeMinor` are the cell-type keys in `sample_sc.h5ad`):
```bash
conda activate tangram_env
python -m reference_aligners.run_tangram \
    --scdata        sample_dataset/scRNA/sample_sc.h5ad \
    --stdata        sample_dataset/ST/sample_st.h5ad \
    --output_folder sample_output/tangram/sample \
    --cell_type_key cellType
```

---

### 4. Run a reference aligner (all pairs)

Runs the chosen aligner for every row in `pairs.csv`, iterating over all cell-type keys
defined in `scRNA/index.csv`. Metrics are computed automatically after each run.

```bash
conda activate tangram_env   # or tacco_env / dot_env
python -m batch_processing.run_reference_aligner_all_pairs \
    --aligner    tangram/tacco/dot \
    --pairs_csv  <path/to/pairs.csv> \
    --sc_dir     <path/to/scRNA> \
    --st_dir     <path/to/ST> \
    --output_dir <output/tangram>
```

**Example with sample dataset:**
```bash
conda activate tangram_env
python -m batch_processing.run_reference_aligner_all_pairs \
    --aligner    tangram \
    --pairs_csv  sample_dataset/pairs.csv \
    --sc_dir     sample_dataset/scRNA \
    --st_dir     sample_dataset/ST \
    --output_dir sample_output/tangram/sample
```

---

### 5. Run the novel method (single pair)

```bash
conda activate alternative_idea_env
python main.py \
    --scdata        <path/to/sc.h5ad> \
    --stdata        <path/to/st.h5ad> \
    --output_folder <output/pair_0> \
    [--K 20] \
    [--lr 0.008] \
    [--epochs 1000] \
    [--leiden_resolution 3.0] \
#    [--sc_embedding_method pca] \
#    [--sc_embedding_d 32] \
    [--lambda_rec_spot 0.5] \
    [--lambda_rec_gene 0.5] \
    [--lambda_state_entropy 0.1] \
    [--lambda_spot_entropy 0.08] \
    [--lambda_soft_contingency 1.0] \
    [--gpu_limit_gb 48] \
    [--logging verbose]
```

Writes to `output_folder/`:
- `gep_prob.h5ad` / `gep_det.h5ad` — probabilistic and deterministic predicted GEPs (G × S)
- `mapping_prob.h5ad` / `mapping_det.h5ad` — spot-to-cell assignment matrices (C × S)
- `loss/` — per-epoch loss curves and final values CSV
- `intermediate/` — B, C, M matrices and state usage JSON (only when `--store_intermediate` is set)
- `analysis/` — post-mapping analysis: UMAP comparison, spatial cell-state plot, state profiles, substate metrics, contingency heatmap in a PDF report (generated with `typst`)

**Example with sample dataset:**
```bash
conda activate alternative_idea_env
python main.py \
    --scdata        sample_dataset/scRNA/sample_sc.h5ad \
    --stdata        sample_dataset/ST/sample_st.h5ad \
    --output_folder sample_output/sample
```

> Please mind that the results from mapping this sample datasets are not to be interpreted.
> This sample dataset is only for syntactical purposes to show how to use this repository.
> There is no biological meaning to this dataset.

---

### 6. Run grid search (single pair)

Hyperparameters are defined in a YAML config. List values create grid axes;
all combinations are executed sequentially.

```bash
conda activate alternative_idea_env
python -m batch_processing.grid_search.grid_search \
    -c              batch_processing/grid_search/grid_search_config.yaml \
    --scdata        <path/to/sc.h5ad> \
    --stdata        <path/to/st.h5ad> \
    --output_folder <output/grid_search/pair_0> \
    [--gpu_limit_gb 48] \
    [--logging verbose]
```

Results are written to numbered subdirectories (`0/`, `1/`, …) with a `summary.csv` at the top level.

**Example with sample dataset:**
```bash
conda activate alternative_idea_env
python -m batch_processing.grid_search.grid_search \
    -c              batch_processing/grid_search/grid_search_config.yaml \
    --scdata        sample_dataset/scRNA/sample_sc.h5ad \
    --stdata        sample_dataset/ST/sample_st.h5ad \
    --output_folder sample_output/grid_search/sample
```

**Config format** (`grid_search_config.yaml`):

```yaml
model:
  K: [10, 20, 40]          # list → grid axis
training:
  lr: 0.008
  epochs: 1000
  reference_leiden_clustering_resolution: 3.0
#sc_embedding:
#  method: pca
#  d: 32
loss_weights:
  lambda_rec_spot: 0.5
  lambda_rec_gene: 0.5
  lambda_state_entropy: 0.1
  lambda_spot_entropy: 0.08
  lambda_soft_contingency: 1.0
```

---

### 7. Run grid search (all pairs, multi-GPU)

Runs the full grid search for every pair in `pairs.csv`. One pair runs per GPU in parallel.

```bash
conda activate alternative_idea_env
python -m batch_processing.grid_search.run_grid_search_all_pairs \
    -c              batch_processing/grid_search/grid_search_config.yaml \
    --pairs_csv     <path/to/pairs.csv> \
    --sc_dir        <path/to/scRNA> \
    --st_dir        <path/to/ST> \
    --output_dir    <output/grid_search> \
    --gpus 0 1 2 3 \
    [--gpu_limit_gb 48]
```

Output layout:

```
<output_dir>/
  configs/          # per-pair YAML configs derived from the template
  pair_0/
    summary.csv
    analysis_overview.csv
    0/, 1/, …       # one subdirectory per grid-search run
  pair_1/
    …
```

**Example with sample dataset:**
```bash
conda activate alternative_idea_env
python -m batch_processing.grid_search.run_grid_search_all_pairs \
    -c           batch_processing/grid_search/grid_search_config.yaml \
    --pairs_csv  sample_dataset/pairs.csv \
    --sc_dir     sample_dataset/scRNA \
    --st_dir     sample_dataset/ST \
    --output_dir output/grid_search/sample \
    --gpus 0
```
