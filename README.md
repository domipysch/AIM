# AIM — Annotation-Independent Mapping


**AIM** maps an **unannotated** scRNA-seq reference (scRNA) onto single-cell-resolution spatial transcriptomics (ST) in a **GUI** or **CLI**. For a scRNA/ST pair it:

1. **Over-clusters** the scRNA reference once with Leiden into `L` start clusters.
2. Builds **one agglomeration tree** over those start clusters.
3. **Sweeps `K`**: for every `K` in `(1,L)` it cuts the tree into `K` cell states and assigns **exactly one** to each spot, reconstructing spot expression from the state profiles.
4. **Finds best `K`**: computes metrics (reconstruction quality, spatial coherence and transcriptional coherence) and helps the user find the most sensible cell state granularity via an interactive GUI.

The spot→state step is **modular**. In principle, any scRNA to ST alignment tool can be used here.
We implement two baseline mappers (`nearest_centroid`, `wann`) and three reference mappers (`tangram`,  `tacco`, `dot`).
You are invited to add and try out your mapper of choice!

If you want to replace the Leiden overclustering from Step 1 with your own annotations,
`--start_from_annotation` takes an existing annotation's `L` cell types as the start clusters instead.

## Outline

- [Why AIM?](#why-aim)
- [Input data format](#input-data-format)
- [For users](#for-users)
- [For developers](#for-developers)

## Why AIM?

Conventional mapping methods blindly map a **pre-annotated** scRNA reference onto spots, even if there is not enough signal in the spatial data to reliably distinguish between them.
The hierarchical cell type clustering is based on both the scRNA and the ST data at hand. The user can find a cell type granularity that can be mapped reliably.

![aim_gui](./docs/aim_gui.gif)

## Input data format

To run AIM, you need a pair of a scRNA dataset and a ST dataset, both as raw counts in `.h5ad` format.

- **scRNA `<Name>.h5ad`**: `X` = raw counts (cells × genes);  `var_names` = **uppercase** gene symbols.
- **ST `<Name>.h5ad`**: `X` = raw counts (spots × genes);  `var_names` = **uppercase** gene symbols; `obsm["spatial"]` = (x, y)-coordinates of the spots: (n_spots × 2).

### Single pair

Just have the two `.h5ad` files available and you are good to go.

### Batch processing

AIM also supports batch processing of multiple pairs.
Organise the data relationally the following way so one scRNA reference can be reused  across many ST slices:

```
DATA/
├── scRNA/
│   ├── index.csv          # one row per scRNA dataset
│   └── <scName>.h5ad
├── ST/
│   ├── index.csv          # one row per ST dataset
│   └── <stName>.h5ad
└── pairs.csv              # links scRNA ↔ ST: PairID, scName, stName, …
```

See [`sample_dataset/`](sample_dataset) for a minimal example, and validate your own layout with [`aim data validate`](#validate-a-dataset-aim-data-validate).

## For users

### Installation

AIM is installable from PyPI:

```bash
pip install spatial-aim
```

This installs one command, `aim`, with subcommands (`aim --help`).
You can now already run `aim` with the baseline mappers.

If you want to be able to select one of the reference aligners within AIM, you need `conda` installed on your machine.
Please install the corresponding conda environments on your computer.

| Method  | Environment   | Command                                                                                                                                                                              |
| ------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Tangram | `tangram_env` | `conda env create -f https://raw.githubusercontent.com/domipysch/AIM/main/src/aim/reference_aligners/environment_tangram.yml`                                                        |
| TACCO   | `tacco_env`   | `conda env create -f https://raw.githubusercontent.com/domipysch/AIM/main/src/aim/reference_aligners/environment_tacco.yml`                                                          |
| DOT     | `dot_env`     | `conda env create -f https://raw.githubusercontent.com/domipysch/AIM/main/src/aim/reference_aligners/environment_dot.yml` then `Rscript -e "remotes::install_github('saezlab/DOT')"` |

> For AIM to use the environments, please make sure AIM can find your conda executable.
> Either set the environment variable `CONDA_EXE` or add your path to your conda installation to `PATH`.
> Make sure `exe = os.environ.get("CONDA_EXE") or shutil.which("conda")` returns the path to your conda.

### Validate a dataset pair (`aim data validate`)

If you have structured your data as shown in section [Batch Processing](#batch-processing), you can check whether the main requirements are met.
It checks every scRNA and ST `.h5ad` against its `index.csv` row and validates all count matrices (raw counts, uppercase gene names, shapes, ST spatial coords present). Exits non-zero on any error.

```bash
aim data validate --data-root path/to/DATA
```

### Run via GUI (`aim gui`)

Interactive [Streamlit](https://streamlit.io) app to run AIM from the browser and interactively browse the results for one scRNA/ST pair without any .

```bash
aim gui [--server_port 8501]
```

Open the printed URL (default http://localhost:8501), set the inputs, pick one or multiple mapper(s), and click **Run**.
The GUI writes each mapper's sweep to  `<output_dir>/<mapper>/`. The results using each mapper are visualized in different tabs. When having run multiple mappers, also a "Compare" tab appears.

### Run via CLI (`aim run`)

Run AIM via CLI on a single pair or in batch mode.
The spot→state mapper is chosen with argument `--mapping`:

- **`nearest_centroid`** (default): Per spot, select the cell state by highest cosine similarity between the state's centroid and the spot's expression. One-hot assignment by default.
- **`wann`**: Weighted Adaptive Nearest Neighbor (Di Salvo et al., TMLR 2025): each reference cell gets a label-reliability score; each spot inherits the neighbourhood size of its nearest reference cell and votes over its nearest neighbours (labelled single reference cells by cosine distance) weighted by reliability.
- **`tangram` / `tacco` / `dot`**: delegate the spot→state step to given external aligner (one alignment per `K`, run in its own conda env; slower). Requires the corresponding reference env (see [Installation](#installation)).

The linkage of the agglomeration tree over the start clusters is chosen with `--agglo_tree_method`:

- **`ward`** (default): Ward's criterion (R's `ward.D`); carries a size term and tends to produce balanced states.
- **`average`**: UPGMA — average pairwise distance; no size term, tends to peel small tight groups off a growing dominant state.

The **start clusters** the tree is built over come from the Leiden over-clustering by default.
Pass `--start_from_annotation <obs_column>` to use a pre-existing annotation instead: its cell types become the start clusters, no over-clustering is computed at all,
and `K` sweeps from the number of annotated types down (cells with no label are dropped).
Give such runs their own `--output_dir`: a run root is named after the mapper alone, so the two modes would otherwise overwrite each other.

**Single pair**

```bash
aim run --scdata path/to/sc.h5ad \
		--stdata path/to/st.h5ad \
		--output_dir path/to/out_dir \
		[--mapping nearest_centroid|wann|tangram|tacco|dot] \
		[--agglo_tree_method ward|average] \
		[--start_from_annotation <obs_column>] \
		[--leiden_resolution 3.0] \
		[--k_min 1] [--k_max <L>] [--k_step 1] \
		[--logging verbose]
```

**Batch mode (all pairs in `pairs.csv`):**

```bash
aim run --pairs_csv path/to/pairs.csv \
		--sc_dir path/to/scRNA \
		--st_dir path/to/ST \
		--output_dir path/to/out_dir \
		[--mapping nearest_centroid|wann|tangram|tacco|dot] \
		[--agglo_tree_method ward|average] \
		[--start_from_annotation <obs_column>] \
		[--leiden_resolution 3.0] \
		[--k_min 1] [--k_max <L>] [--k_step 1] \
		[--logging verbose]
```

Each pair is written to `<out_dir>/<PairID>_<scName>__<stName>/`.

**Output layout** (per pair):

- `<mapping>/config.yaml` — the run configuration.
- `start_clustering.h5ad` — per-cell start cluster (`start_cluster` id + `start_cluster_name`); computed once, reused for every `K`.
- `k_<kkk>/` — one folder per `K`:
	- `spot_to_state_mapping_soft.h5ad` — the spot→state matrix `P` (spots × `K`);
	    carries `obs["mapping_confidence"]` when the mapper defines one.
	- `spot_to_state_mapping.csv` — `P` as CSV (for eyeballing).
	- `start_cluster_to_state.csv` — the start-cluster→state tree cut.
	- `analysis/data/` — machine-readable metrics for that `K` (no figures; the GUI renders them).

**Sample-dataset example:**

```bash
aim run \
	--scdata     sample_dataset/scRNA/sample_sc.h5ad \
	--stdata     sample_dataset/ST/sample_st.h5ad \
	--output_dir sample_output/sample
```

> The sample dataset is for sample usage only, it has no meaning.

### Annotation-based baseline

There is no separate command for it: `aim run --start_from_annotation <obs_column>` covers it.
The annotated types become the start clusters, and the sweep's **`K` = number-of-types** level is the baseline.

## For developers

### Installation of dev environment

If you want to contribute, clone this repository and create a local conda environment.
This will include everything apart from the reference aligner methods.

```bash
conda env create -f environment.yml
conda activate aim_env

# Optional: Install the AIM-CLI from local
pip install -e . --no-deps
```

#### Environments of reference aligners

If you want to use one of the reference aligners within AIM or standalone, you have to additionally create the following environments. Each reference aligner runs in its own conda environment.

AIM orchestrates them out-of-process via `conda run`, so you need `conda` on `PATH`. The env `.yml` files live in the repo under [`src/aim/reference_aligners/`](https://github.com/domipysch/AIM/tree/main/src/aim/reference_aligners).

| Method  | Environment   | Create                                                                                                                                                                               |
| ------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Tangram | `tangram_env` | `conda env create -f https://raw.githubusercontent.com/domipysch/AIM/main/src/aim/reference_aligners/environment_tangram.yml`                                                        |
| TACCO   | `tacco_env`   | `conda env create -f https://raw.githubusercontent.com/domipysch/AIM/main/src/aim/reference_aligners/environment_tacco.yml`                                                          |
| DOT     | `dot_env`     | `conda env create -f https://raw.githubusercontent.com/domipysch/AIM/main/src/aim/reference_aligners/environment_dot.yml` then `Rscript -e "remotes::install_github('saezlab/DOT')"` |


### Adding another reference aligner

Reference aligners are pluggable.
Each runs the same way: a wrapper launched out-of-process in its own conda env.
The framework learns about it from a **single registry**,[`src/aim/reference_aligners/registry.py`](src/aim/reference_aligners/registry.py).

Adding a reference aligner is three steps:

**1. Add the conda environment file and create it**, e.g. `conda env create -f src/aim/reference_aligners/environment_<name>.yml`.

**2. Write `src/aim/reference_aligners/run_<name>.py`** satisfying the CLI API:

|            | Contract                                                                                                                                            |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Args**   | `--scdata`, `--stdata`, `--output_folder`, `--cell_type_key`                                                                                        |
| **Input**  | `scdata` carries the states/types in `obs[cell_type_key]`; `stdata` shares genes with `scdata` and has spatial coords in `obsm["spatial"]`          |
| **Output** | writes `<output_folder>/mapping_prob.h5ad` — AnnData with `obs` = spots (ST order), `var` = state/type names, `X` = float32 soft-assignment (S × T) |

Invoked with only those four args, the wrapper must reproduce its canonical mapping.

**3. Register it**

Add one line to `REFERENCE_ALIGNERS` ([`src/aim/reference_aligners/registry.py`](src/aim/reference_aligners/registry.py)).

```python
ReferenceAligner("<name>", "<name>_env", "aim.reference_aligners.run_<name>"),
```

It is then selectable everywhere automatically:

- `aim gui` (it will be selectable in the sidebar)
- `aim run --mapping <name>` (as an AIM reference mapper, one alignment per `K`) — including `--start_from_annotation`, where it maps onto a pre-existing annotation

**4. Create PR**

Create a PR with those changes to make it available for all users!

