# AIM — Annotation-Independent Mapping

AIM maps an scRNA-seq reference onto single-cell-resolution spatial
transcriptomics (ST) without being told a cell-type granularity up front.

**AIM is the whole framework, not a single mapper.** For a scRNA/ST pair it:

1. **Over-clusters** the reference once with Leiden into `L` subclusters.
2. Builds **one agglomeration tree** over those subclusters.
3. **Sweeps `K`**: for every `K` it cuts the tree into `K` cell states and maps
   each ST spot onto them, reconstructing spot expression from the state
   profiles.

The spot→state step is **modular** — it can be one of AIM's own mappers
(`nearest_centroid`, `wann`) or an external reference aligner (`tangram`,
`tacco`, `dot`) run once per `K`. Because `K` is *swept* rather than chosen, one
run produces the whole granularity spectrum; the interactive GUI is used to
browse it.

- [Input data](#input-data)
- [Installation](#installation)
- [Environments](#environments)
- [Usage](#usage)
  - [GUI](#gui-aim-gui)
  - [AIM sweep — single & batch](#aim-sweep-aim-run)
  - [Reference aligner — single & batch](#reference-aligner-aim-run-reference)
  - [Validate a dataset](#validate-a-dataset-aim-data-validate)
- [Adding a reference aligner](#adding-a-reference-aligner)

---

## Input data

You need pairs of an scRNA-seq dataset and a spatial dataset, both as `.h5ad`
(raw counts preferred, though pre-processed data also works).

- **scRNA `<Name>.h5ad`** — `X` = raw counts (cells × genes, float32);
  `var_names` = **uppercase** gene symbols; `obs` has at least one cell-type
  column.
- **ST `<Name>.h5ad`** — `X` = raw counts (spots × genes, float32);
  `var_names` = **uppercase** gene symbols; `obsm["spatial"]` = float array
  (n_spots × 2).

**Single pair** — just have the two `.h5ad` files.

**Batch** — organise the pairs relationally so one scRNA reference can be reused
across many ST slices:

```
dataset/
├── scRNA/
│   ├── index.csv          # one row per scRNA dataset (incl. CellTypeKey0/1/2)
│   └── <scName>.h5ad
├── ST/
│   ├── index.csv          # one row per ST dataset
│   └── <stName>.h5ad
└── pairs.csv              # links scRNA ↔ ST: PairID, scName, stName, …
```

See [`sample_dataset/`](sample_dataset) for a minimal, runnable example, and
validate your own layout with [`aim data validate`](#validate-a-dataset-aim-data-validate).

---

## Installation

AIM's core (`aim run`) is installable from PyPI:

```bash
pip install spatial-aim            # core: aim run + aim data validate
pip install "spatial-aim[gui]"     # + the Streamlit results GUI (aim gui)
```

> **While the package is on TestPyPI**, pull the package from TestPyPI but its
> dependencies from PyPI:
> ```bash
> pip install --index-url https://test.pypi.org/simple/ \
>             --extra-index-url https://pypi.org/simple/ spatial-aim
> ```

This installs one command, `aim`, with subcommands (`aim --help`).

**Recommended (conda), for the GUI and reproducibility.** The core pulls heavy
scientific deps (torch, scanpy, squidpy); the pinned conda environment is the
tested setup:

```bash
conda env create -f environment.yml     # creates aim_env with everything
conda activate aim_env
pip install spatial-aim --no-deps        # or: pip install -e . --no-deps (from a checkout)
```

The reference aligners are **not** installed by any of the above — they live in
their own conda environments (see below).

---

## Environments

Each reference aligner runs in its own conda environment; AIM orchestrates them
out-of-process via `conda run`, so you only need `conda` on `PATH` (no manual
activation). The env `.yml` files ship with the package and live in the repo
under `src/aim/reference_aligners/`.

| Method | Environment | Create |
|--------|-------------|--------|
| AIM (core + GUI) | `aim_env` | `conda env create -f environment.yml` |
| Tangram | `tangram_env` | `conda env create -f src/aim/reference_aligners/environment_tangram.yml` |
| TACCO | `tacco_env` | `conda env create -f src/aim/reference_aligners/environment_tacco.yml` |
| DOT | `dot_env` | `conda env create -f src/aim/reference_aligners/environment_dot.yml` then `Rscript -e "remotes::install_github('saezlab/DOT')"` |

**Make `aim` importable inside each reference env** so its wrapper module can be
launched there:

```bash
conda run -n tangram_env pip install --no-deps spatial-aim   # and tacco_env / dot_env
```

`--no-deps` is deliberate: the aligner's heavy deps already live in its env, and
AIM's core deps must not be pulled in.

---

## Usage

Everything is a subcommand of `aim`. Reference-aligner commands are run from
`aim_env`; the aligner itself executes in its own env automatically.

### GUI (`aim gui`)

Interactive [Streamlit](https://streamlit.io) app to run and browse an AIM sweep
for one sc/ST pair. Everything — the sc/ST paths, output directory, K range,
linkage, and mapper — is configured in the sidebar; the only CLI argument is the
port.

```bash
conda activate aim_env
aim gui                         # then configure & Run in the sidebar
aim gui --server_port 8600
```

Open the printed URL (default http://localhost:8501), set the inputs, pick a
mapper, and click **Run**. The GUI writes each mapper's sweep to
`<output_dir>/<mapper>/` and renders all plots on demand from the per-K metrics.
Mappers without a per-spot confidence (`tangram`, `tacco`, `dot`) disable the
confidence slider.

### AIM sweep (`aim run`)

Runs the full over-cluster → agglomerate → per-K mapping sweep. The spot→state
mapper is chosen with `--mapping`:

- **`nearest_centroid`** (default) — zero-parameter nearest-centroid by cosine.
  One-hot assignment, so soft and deterministic reconstructions coincide.
- **`wann`** — Weighted Adaptive Nearest Neighbor (Gallo et al., TMLR 2025):
  each reference cell gets a label-reliability score; each spot inherits the
  neighbourhood size of its nearest reference cell and votes over its neighbours
  weighted by reliability. Parameter-free.
- **`tangram` / `tacco` / `dot`** — delegate the spot→state step to that external
  aligner (one alignment per `K`, run in its own conda env; slower). Requires the
  corresponding reference env (see [Environments](#environments)).

**Single pair:**

```bash
conda activate aim_env
aim run \
    --scdata     path/to/sc.h5ad \
    --stdata     path/to/st.h5ad \
    --output_dir out/pair_0 \
    [--mapping nearest_centroid|wann|tangram|tacco|dot] \
    [--leiden_resolution 3.0] \
    [--k_min 1] [--k_max <L>] [--k_step 1] \
    [--logging verbose]
```

> `K` is swept, not set: the run covers every `K` in `[k_min, k_max]` (default
> `1 … L`, where `L` = number of Leiden clusters at `--leiden_resolution`). There
> is no `--K` argument.

**Batch (all pairs in `pairs.csv`):**

```bash
conda activate aim_env
aim run \
    --pairs_csv  path/to/pairs.csv \
    --sc_dir     path/to/scRNA \
    --st_dir     path/to/ST \
    --output_dir out/aim \
    [--mapping … --leiden_resolution … --k_min … --k_max … --k_step …]
```

Each pair is written to `<output_dir>/<PairID>_<scName>__<stName>/`.

**Output layout** (per pair):

- `<mapping>/config.yaml` — the run configuration.
- `leiden_overclustering.h5ad` — per-cell Leiden label; computed once, reused for every `K`.
- `k_<kkk>/` — one folder per `K`:
  - `spot_to_state_mapping_soft.h5ad` — the spot→state matrix `P` (spots × `K`);
    carries `obs["mapping_confidence"]` when the mapper defines one.
  - `spot_to_state_mapping.csv` — `P` as CSV (for eyeballing).
  - `leiden_to_state.csv` — the subcluster→state tree cut.
  - `analysis/data/` — machine-readable metrics for that `K` (no figures; the GUI renders them).

**Sample-dataset example:**

```bash
conda activate aim_env
aim run \
    --scdata     sample_dataset/scRNA/sample_sc.h5ad \
    --stdata     sample_dataset/ST/sample_st.h5ad \
    --output_dir sample_output/sample
```

> The sample dataset is for showing usage only — its results carry no biological meaning.

### Reference aligner (`aim run-reference`)

Runs a reference aligner (Tangram / TACCO / DOT) **directly** (not per-K inside a
sweep) with its canonical baseline settings, then computes the mapping metrics.
Run from `aim_env`; the aligner executes in its own env via `conda run`.

**Single pair** — give the cell-type key (an `obs` column of the scRNA data):

```bash
conda activate aim_env
aim run-reference --aligner tangram \
    --scdata        path/to/sc.h5ad \
    --stdata        path/to/st.h5ad \
    --output_dir    out/tangram/pair_0 \
    --cell_type_key cellType
```

**Batch** — one subtree per pair × cell-type granularity (the keys come from
`scRNA/index.csv`'s `CellTypeKey0/1/2`):

```bash
conda activate aim_env
aim run-reference --aligner tangram \
    --pairs_csv  path/to/pairs.csv \
    --sc_dir     path/to/scRNA \
    --st_dir     path/to/ST \
    --output_dir out/tangram
```

Each run writes `mapping_prob.h5ad` (a spots × cell-types soft-assignment
AnnData) and `analysis/data/` to its output folder.

### Validate a dataset (`aim data validate`)

Checks every scRNA and ST `.h5ad` against its `index.csv` row, then validates all
pairs (raw counts, uppercase gene names, shapes, spatial coords). Exits non-zero
on any error.

```bash
aim data validate --data-root path/to/dataset
```

---

## Adding a reference aligner

Reference aligners are pluggable. Each runs the same way — a small wrapper
launched out-of-process in its own conda env — and the whole framework learns
about it from a **single registry**,
[`src/aim/reference_aligners/registry.py`](src/aim/reference_aligners/registry.py).
Adding one is three steps:

**1. Create its conda env**, e.g. `conda env create -f src/aim/reference_aligners/environment_<name>.yml`.

**2. Write `src/aim/reference_aligners/run_<name>.py`** satisfying the CLI contract:

| | Contract |
|---|---|
| **Args** | `--scdata`, `--stdata`, `--output_folder`, `--cell_type_key` |
| **Input** | `scdata` carries the states/types in `obs[cell_type_key]`; `stdata` shares genes with `scdata` and has spatial coords in `obsm["spatial"]` |
| **Output** | writes `<output_folder>/mapping_prob.h5ad` — AnnData with `obs` = spots (ST order), `var` = state/type names, `X` = float32 soft-assignment (S × T) |

Invoked with only those four args, the wrapper must reproduce its canonical
mapping (bake any other choices into the CLI defaults).

**3. Register it** — add one line to `REFERENCE_ALIGNERS`:

```python
ReferenceAligner("<name>", "<name>_env", "aim.reference_aligners.run_<name>"),
```

It is then selectable everywhere automatically: `aim run --mapping <name>` (as an
AIM reference mapper, one alignment per `K`), `aim run-reference --aligner <name>`
(single/batch), and `ReferenceMapper`. The single runner `run_aligner(...)` in the
registry is the only code path that executes an aligner, so no bespoke dispatch is
needed.
