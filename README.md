# Spatial Transcriptomics Alignment

Alignment of scRNA-seq data with high-res spatial transcriptomics data.

## Overview

This repository contains code for scRNA-seq to spatial transcriptomics alignment.
The task consists of inferring gene expression at spatially resolved locations by mapping scRNA-seq reference data to spatial transcriptomics (ST) spots.

It consists of two main components:

- an evaluation framework with metrics for four objectives in `src/metrics`.
- a noval alignment method in `src/alternative_idea` that does not learn a cell-to-spot mapping directly, but instead also learns a cell-to-cell-state mapping, effectively mapping those cell-states to spots.

## Repository Structure

```
src/
├── aligners/           # Wrappers for baseline methods (Tangram, TACCO, DOT, CytoSPACE)
├── alternative_idea/   # Novel alignment method
├── metrics/            # Evaluation metrics for O1, O2, O4
├── data_preparation/   # Dataset preprocessing scripts
├── utils/              # Shared I/O utilities
└── run_experiment.py   # Experiment orchestrator (grid search over hyperparameters)
```

## Dataset Format

Each dataset folder must contain two h5ad files (see `sample_dataset/` for a reference example).

### `sc.h5ad` — scRNA-seq data

| Slot | Content |
|------|---------|
| `X` | Expression matrix (cells × genes, float32) |
| `obs` | Cell metadata — index: `cellID`, columns include `cellType`, `cellTypeMinor` |
| `var` | Gene metadata — index: `geneID` |

### `st.h5ad` — Spatial transcriptomics data

| Slot | Content |
|------|---------|
| `X` | Expression matrix (spots × genes, float32) |
| `obs` | Spot metadata — index: `spotID` |
| `var` | Gene metadata — index: `geneID` |
| `obsm["spatial"]` | 2D array coordinates, shape (n_spots × 2): `[cArray0, cArray1]` |

### Migrating from the legacy CSV format

If you have existing datasets in the old 6-CSV format, convert them with:

```bash
conda activate alternative_idea_env

# Convert a single dataset folder:
python -m src.data_preparation.convert_csv_to_h5ad -d <dataset_folder>

# Convert all subdirectories under a root folder:
python -m src.data_preparation.convert_csv_to_h5ad -d <root_folder> --all
```

## Installation & Running

Since every alignment method has its own dependencies and requirements,
different conda environments have to be created for each method.

### AlternativeIdea

```bash
conda env create -f src/alternative_idea/environment.yml
conda activate alternative_idea_env
```

### Tangram

```bash
conda env create -f src/aligners/environment_tangram.yml
conda activate tangram_env
```

### TACCO

```bash
conda env create -f src/aligners/environment_tacco.yml
conda activate tacco_env
```

### DOT

DOTr is not available on conda-forge and must be installed from GitHub after creating the environment:

```bash
conda env create -f src/aligners/environment_dot.yml
conda activate dot_env
Rscript -e "remotes::install_github('saezlab/DOT')"
```

## Usage

All commands are run from the root directory of this repository.

### Running a Baseline Aligner

```bash
conda activate tangram_env  # or tacco_env / dot_env
python -m src.aligners.run_tangram \
  -d <dataset_folder> \
  -o <output_path.csv>
```

Each aligner script follows the same pattern (`run_tacco.py`, `run_dot.py`).
For more detailed arguments and options, please refer to the respective script.

Example using `sample_dataset/`:

```bash
conda activate tangram_env
python -m src.aligners.run_tangram \
  -d sample_dataset \
  -o sample_dataset/results/tangram_GEP.csv

conda activate tacco_env
python -m src.aligners.run_tacco \
  -d sample_dataset \
  -o sample_dataset/results/tacco_GEP.csv

conda activate dot_env
python -m src.aligners.run_dot \
  -d sample_dataset \
  -o sample_dataset/results/dot_GEP.csv
```

### Computing Metrics

```bash
python -m src.metrics.run_all_metrics \
  -d <dataset_folder> \
  -r <result_gep.csv> \
  -m <metrics_output_folder>
```

Example using `sample_dataset/results/tangram_GEP.csv`:

```bash
python -m src.metrics.run_all_metrics \
  -d sample_dataset \
  -r sample_dataset/results/tangram_GEP.csv \
  -m sample_dataset/metrics
```

### Running our novel method

Dataset paths and output folders are configured directly in the experiment YAML (see `experiment_config.yaml`):

```yaml
data:
  sc_path: /path/to/sc.h5ad
  st_paths:
    - /path/to/DatasetA/st.h5ad
    - /path/to/DatasetB/st.h5ad  # optional: run once per dataset

output:
  result_folder: /path/to/results
  metric_folder: /path/to/metrics
```

Then run:

```bash
conda activate alternative_idea_env

python -m run_experiment \
  -c experiment_config.yaml \
  [--save_result] \
  [--run_permutation_tests] \
  [--logging verbose]
```

The experiment config supports grid search: list values become search axes and the runner executes every combination.
When multiple `st_paths` are specified, the full grid search runs once per dataset.
Results are written to `<result_folder>/<dataset_name>/<run_id>/` with a `summary.csv` per dataset.

## TODO

- Tests
- Build installable package when ready (maybe extract only our alignment method code for that? or also with evaluation?)
