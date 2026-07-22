"""
AIM — Annotation Independent Mapping.

Maps an scRNA reference onto spatial transcriptomics (ST) spots by
over-clustering the reference, agglomerating the subclusters into K cell states,
and assigning every ST spot to those states:

    1. Leiden over-cluster the reference -> L subclusters (+ centroids, sizes).
    2. Build the agglomeration tree ONCE with average-linkage on the shared-gene
       cosine distance between subcluster centroids (scipy.linkage).
    3. For every K from L down to 1: cut the tree at K states, assemble the
       (size-weighted) merged state profiles M, map every ST spot onto those
       states, and run the post-mapping analysis for that K. The hard tree cut
       is the cluster->state map; only the spot->state map P varies with K.

The two halves of the method live in separate modules:

    clustering half — aggregation.py (per-subcluster / per-state profiles)
                      tree.py        (build tree, cut at K -> labels_k)
    mapping half    — mapping/       (unified SpotStateMapper API + greedy/learned)

with the sweep in sweep.py, disk I/O in io.py, and plots in plots.py. The run
configuration (``AIMConfig``) and the mapper registry / ``build_mapper`` factory
live in aim_config.py; the single-pair / batch drivers that consume an AIMConfig
live in ``main.py`` at the repository root, next to the CLI.

The spot->state mapping step is modular (choose with --mapping):

    greedy  (default) — zero-parameter nearest-centroid classifier. Each spot is
                        assigned to the state whose (size-weighted) centroid is
                        most cosine-similar to it on the shared genes. P is
                        one-hot, so the soft and deterministic reconstructions
                        coincide.
    learned           — a soft P learned by gradient descent (mapping/learned.py),
                        minimizing spot-wise + gene-wise cosine distance with a
                        quadratic spot_gini sharpener (optional warmup).

Each K's folder is written in the exact layout the post-mapping analysis expects
(spot_to_state_mapping.h5ad = P, leiden_to_state.csv = the subcluster->state cut,
plus the run-root leiden_overclustering.h5ad and config.yaml), so the sweep calls
analysis.analysis.run_analysis once per K folder, right after writing it.

The command-line entry point is ``main.py`` at the repository root.
"""

from .aim_config import MAPPING_CHOICES, AIMConfig
from .mapping import SpotStateMapper
from .sweep import run

__all__ = [
    "AIMConfig",
    "MAPPING_CHOICES",
    "run",
    "SpotStateMapper",
]
