"""Post-mapping analysis, decoupled from both the AIM runner (main.py) and
grid_search.py: reads a run's saved mapping outputs from disk and produces the
analysis report from them. Neither main.py nor grid_search.py compute the
analysis matrices inline anymore — both call into this module instead.

Standalone usage (after `python main.py` has written its outputs):
    python -m evaluate_k.run_from_output \\
        --scdata sc.h5ad --stdata st.h5ad --output_folder <run_dir> \\
        --leiden_resolution 3.0
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
from anndata import AnnData

from evaluate_k.analysis import run_analysis
from evaluate_k.report import generate_per_k_report

logger = logging.getLogger(__name__)


def load_mapping_outputs(output_folder: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load the cell->state matrix B and spot->state matrix C from an AIM run's
    saved outputs.

    B = G[leiden_labels] (C x L) is reconstructed from clusters_prob.h5ad
    (obs["leiden_cluster"] — each cell's hard Leiden overclustering label) and
    leiden_merge_prob.h5ad (G, the Leiden-subcluster -> computed-state merge
    matrix). C (S x L) is read directly from mapping_prob.h5ad, already in
    spots x states layout. Both are threshold-zeroed below 0.1, matching the
    convention run_analysis expects.

    Args:
        output_folder: Folder containing mapping_prob.h5ad, leiden_merge_prob.h5ad,
                        and clusters_prob.h5ad, as written by main.py.

    Returns:
        B: cell -> state matrix (C x L).
        C: spot -> state matrix (S x L).
        K: number of computed states.
    """
    output_folder = Path(output_folder)
    clusters_path = output_folder / "clusters_prob.h5ad"
    leiden_merge_path = output_folder / "leiden_merge_prob.h5ad"
    mapping_path = output_folder / "mapping_prob.h5ad"
    for path in (clusters_path, leiden_merge_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(f"Required mapping output missing: {path}")

    leiden_cluster_names = ad.read_h5ad(clusters_path).obs["leiden_cluster"].to_numpy()
    leiden_idx = np.array(
        [int(name.rsplit("_", 1)[-1]) for name in leiden_cluster_names]
    )

    G = np.asarray(ad.read_h5ad(leiden_merge_path).X)  # (L x L) leiden -> state
    B = G[leiden_idx].astype(np.float32)  # (C x L) cell -> state
    C = np.asarray(ad.read_h5ad(mapping_path).X).astype(np.float32)  # (S x L)
    B[B < 0.1] = 0.0
    C[C < 0.1] = 0.0

    K = int(G.shape[1])
    return B, C, K


def analyze_run(
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
    leiden_resolution: float,
    run_id: str = "0",
    adata_sc: AnnData | None = None,
    adata_st: AnnData | None = None,
    **analysis_kwargs,
) -> tuple[dict, int]:
    """
    Load one run's saved mapping outputs and run the post-mapping analysis +
    per-K report.

    Args:
        sc_path, st_path: Full paths to the sc/st h5ad used for the run. Used to
                           load adata_sc/adata_st unless already-loaded copies are
                           passed in directly (grid_search reuses one loaded copy
                           across many runs instead of re-reading per run).
        output_folder:    Folder containing mapping_prob.h5ad, leiden_merge_prob.h5ad,
                           clusters_prob.h5ad (as written by main.py). analysis/ is
                           written inside this folder.
        leiden_resolution: Leiden resolution used for the run's reference clustering.
        run_id:            Identifier used in the per-K report filename/title.
        adata_sc, adata_st: Optional already-loaded AnnData, to skip re-reading
                           sc_path/st_path.
        **analysis_kwargs: Forwarded to run_analysis — e.g. precomputed
                           adata_processed_base / leiden_labels / leiden_shared_labels,
                           used by grid_search to avoid recomputing PCA/UMAP/Leiden
                           once per K value instead of once per run.

    Returns:
        (results, K): the dict returned by run_analysis, and the number of
        computed-state slots (as opposed to results["n_computed_states"], the
        count of states actually used).
    """
    output_folder = Path(output_folder)
    if adata_sc is None:
        adata_sc = ad.read_h5ad(sc_path)
    if adata_st is None:
        adata_st = ad.read_h5ad(st_path)

    B, C, K = load_mapping_outputs(output_folder)

    analysis_dir = output_folder / "analysis"
    results = run_analysis(
        adata_sc=adata_sc,
        adata_st=adata_st,
        B=B,
        C=C,
        output_dir=analysis_dir,
        K=K,
        leiden_resolution=leiden_resolution,
        **analysis_kwargs,
    )
    generate_per_k_report(analysis_dir, K, run_id)
    logger.info("Analysis report written to %s", analysis_dir)
    return results, K


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Run post-mapping analysis on an existing AIM output folder "
        "(reads mapping_prob.h5ad / leiden_merge_prob.h5ad / clusters_prob.h5ad)."
    )
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Folder containing the AIM run's saved mapping outputs; analysis/ is written here",
    )
    parser.add_argument(
        "--leiden_resolution",
        type=float,
        default=3.0,
        help="Leiden resolution used for the run's reference clustering",
    )
    parser.add_argument(
        "--run_id", type=str, default="0", help="Identifier for the per-K report"
    )
    args = parser.parse_args()

    analyze_run(
        args.scdata,
        args.stdata,
        args.output_folder,
        leiden_resolution=args.leiden_resolution,
        run_id=args.run_id,
    )
