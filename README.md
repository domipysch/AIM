# Spatial Transcriptomics Alignment

Maps scRNA-seq reference data onto high-resolution spatial transcriptomics (ST) spots: Leiden over-clusters the reference, one agglomeration tree merges the subclusters into `K` cell states, and every ST spot is assigned to those states. The spot→state step is modular (`greedy` nearest-centroid or `learned` soft assignment).

## Repository Structure

```
main.py                          # AIM CLI + single-pair / batch drivers
environment.yml                  # Conda env for AIM
src/
├── adata_schema.py              # Canonical obs/var/uns/obsm/obsp/layers key names for the sc/st AnnData objects
├── aim/                         # AIM method — agglomerative K-sweep
│   ├── aim_config.py            #   AIMConfig + mapper registry / build_mapper factory
│   ├── clustering.py            #   clustering half: Leiden over-clustering (all genes + shared genes)
│   ├── aggregation.py           #   clustering half: per-subcluster / per-state profiles
│   ├── tree.py                  #   clustering half: agglomeration tree + cut at K -> labels_k
│   ├── mapping/                 #   mapping half: unified SpotStateMapper API (greedy / learned)
│   ├── io.py                    #   per-K disk outputs (h5ad + CSV)
│   └── sweep.py                 #   the K-sweep orchestration
├── analysis/                    # Post-mapping analysis: orchestration + loaders + typst PDF report
├── metrics/                     # Evaluation metrics (cosine reconstruction, one-hotness, spatial/biology, modularity)
└── plots/                       # Matplotlib figure generation shared across the analyses
reference_aligners/              # Baseline method wrappers (Tangram, TACCO, DOT)
batch_processing/
├── run_pre_check_all_pairs.py   # Batch pre-alignment checks
└── run_reference_aligner_all_pairs.py  # Batch baseline aligners
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
| Novel method | `aim_env` | `conda env create -f environment.yml` |
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
conda activate aim_env
python -m pre_check \
    --scdata        <path/to/sc.h5ad> \
    --stdata        <path/to/st.h5ad> \
    --output_folder <output/pre_check/pair_0> \
    [--leiden_resolution 0.5]
```

**Example with sample dataset:**
```bash
conda activate aim_env
python -m pre_check \
    --scdata        sample_dataset/scRNA/sample_sc.h5ad \
    --stdata        sample_dataset/ST/sample_st.h5ad \
    --output_folder sample_output/pre_check/sample
```

---

### 2. Pre-alignment compatibility check (all pairs)

Runs the pre-check for every row in `pairs.csv` in parallel.

```bash
conda activate aim_env
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
conda activate aim_env
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

The method Leiden over-clusters the reference into `L` subclusters, builds one
agglomeration tree (average-linkage on shared-gene cosine distance), and for
every `K` from `L` down to `1` cuts the tree into `K` cell states and maps each
ST spot onto them. The spot→state mapping is **modular** (`--mapping`):

- **`greedy`** (default) — zero-parameter nearest-centroid: each spot is assigned
  to the state whose profile is most cosine-similar to it. No training; the
  assignment is one-hot, so soft and deterministic reconstructions coincide.
- **`learned`** — a soft `P` trained by gradient descent, minimizing spot-wise +
  gene-wise cosine distance with a quadratic `spot_gini` sharpener (optional
  warmup). Adds `--epochs / --lr / --lambda_spot_gini / --spot_gini_warmup_frac`.

```bash
conda activate aim_env
python main.py \
    --scdata        <path/to/sc.h5ad> \
    --stdata        <path/to/st.h5ad> \
    --output_dir    <output/pair_0> \
    [--mapping greedy|learned] \
    [--leiden_resolution 3.0] \
    [--normalize_and_log] \
    [--k_min 1] [--k_max <L>] [--k_step 1] \
    [--logging verbose] \
    # learned-mode only:
    [--epochs 400] [--lr 0.02] \
    [--lambda_spot_gini 1.0] [--spot_gini_warmup_frac 0.5]
```

> `K` is not a single value — the run sweeps every `K` in `[k_min, k_max]` (default
> `1 … L`, where `L` = number of Leiden clusters at `--leiden_resolution`). There
> is no `--K` argument.

Writes to `output_dir/`:
- `config.yaml` — the run configuration (mapping choice, hyperparameters, `K` range).
- `leiden_overclustering.h5ad` — per-cell Leiden over-cluster label; written once and
  reused by every `K`.
- `k_<kkk>/` — one folder per `K`, in the layout the post-mapping analysis consumes:
  - `spot_to_state_mapping.h5ad` — the spot→state matrix `P` (spots × `K`).
  - `spot_to_state_mapping.csv` — `P` as CSV (tiny values zeroed, rounded) for eyeballing.
  - `leiden_to_state.csv` — the subcluster→state tree cut (`labels_k`).
  - `analysis/` — the post-mapping analysis for that `K`: `report.pdf` (typst) plus
    `plots/` and `data/`. The post-mapping analysis runs for every `K`.

**Example with sample dataset:**
```bash
conda activate aim_env
python main.py \
    --scdata        sample_dataset/scRNA/sample_sc.h5ad \
    --stdata        sample_dataset/ST/sample_st.h5ad \
    --output_dir    sample_output/sample
```

> Please mind that the results from mapping this sample datasets are not to be interpreted.
> This sample dataset is only for syntactical purposes to show how to use this repository.
> There is no biological meaning to this dataset.

---

### 6. Run the novel method (all pairs)

`main.py` also runs every row in `pairs.csv` sequentially — pass the batch flags
instead of the single-pair ones. Each pair is written to
`<output_dir>/<PairID>_<scName>__<stName>/` in the same layout as above.

```bash
conda activate aim_env
python main.py \
    --pairs_csv  <path/to/pairs.csv> \
    --sc_dir     <path/to/scRNA> \
    --st_dir     <path/to/ST> \
    --output_dir <output/agglomerative> \
    [--mapping greedy|learned] \
    [--leiden_resolution 3.0] [--normalize_and_log] \
    [--k_min 1] [--k_max <L>] [--k_step 1]
```

**Example with sample dataset:**
```bash
conda activate aim_env
python main.py \
    --pairs_csv  sample_dataset/pairs.csv \
    --sc_dir     sample_dataset/scRNA \
    --st_dir     sample_dataset/ST \
    --output_dir sample_output/agglomerative/sample
```
