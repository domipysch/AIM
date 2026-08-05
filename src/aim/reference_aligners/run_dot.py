import subprocess
import argparse
import os
import shutil
import logging
import numpy as np
import pandas as pd
from anndata import AnnData
from pathlib import Path

R_SCRIPT = os.path.join(os.path.dirname(__file__), "run_dot.R")

logger = logging.getLogger(__name__)


def _find_rscript():
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    fallback = "/opt/miniconda3/envs/dot_env/bin/Rscript"
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        "Rscript not found. Activate the dot_env conda environment or add it to PATH."
    )


def dot_align_data(
    sc_path: Path,
    st_path: Path,
    cell_type_key: str,
    output_folder: Path,
):
    """
    Run DOT alignment by calling run_dot.R via Rscript.
    Saves mapping_prob.h5ad (obs=spots, var=cell types) to output_folder.

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        cell_type_key: obs column to use as annotation; pass "cellID" to map individual cells.
        output_folder: Folder where the output is written.

    Returns:
        AnnData with obs=spots, var=cell types (S x T layout).
    """
    annotation_key = cell_type_key if cell_type_key else "cellID"

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    map_prob_csv = output_folder / "mapping_prob.csv"

    cmd = [
        _find_rscript(),
        R_SCRIPT,
        str(sc_path),
        str(st_path),
        annotation_key,
        str(map_prob_csv),
    ]
    logger.info("Running DOT via R: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # R writes weights as S x T (rows=spots, cols=cell types, already named) —
    # this is already the layout we want, no transpose needed.
    df = pd.read_csv(map_prob_csv, header=0, index_col=0)
    ad_map = AnnData(
        X=np.asarray(df.values, dtype=np.float32),
        obs=pd.DataFrame(index=df.index.astype(str)),
        var=pd.DataFrame(index=df.columns.astype(str)),
    )
    ad_map.write_h5ad(output_folder / "mapping_prob.h5ad")
    map_prob_csv.unlink(missing_ok=True)

    logger.info("Saved mapping to %s", output_folder / "mapping_prob.h5ad")


def _serve(sc_path: Path, st_path: Path, control_dir: Path) -> None:
    """Map one K per job. DOT's heavy work is in R (re-launched per job), so the
    worker only saves the per-K conda/Python cold start, but keeps the uniform
    AlignerWorker interface used by ReferenceMapper."""
    from aim.reference_aligners.registry import serve_loop

    def handle_job(cell_type_key: str, output_folder: Path) -> None:
        dot_align_data(
            sc_path,
            st_path,
            cell_type_key=cell_type_key,
            output_folder=output_folder,
        )

    serve_loop(control_dir, handle_job)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run DOT alignment via run_dot.R")
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        help="Folder where to store results (required unless --server)",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        required=False,
        default="cellType",
        help="What cell type key to load from sc data as cell type annotation to be mapped.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Persistent-worker mode: map one K per job from --control_dir "
        "(see reference_aligners.registry.AlignerWorker).",
    )
    parser.add_argument(
        "--control_dir",
        type=Path,
        default=None,
        help="Job-queue directory (required with --server)",
    )
    args = parser.parse_args()

    if args.server:
        if args.control_dir is None:
            parser.error("--server requires --control_dir")
        _serve(args.scdata, args.stdata, args.control_dir)
    else:
        if args.output_folder is None:
            parser.error("--output_folder is required")
        dot_align_data(
            args.scdata,
            args.stdata,
            cell_type_key=args.cell_type_key,
            output_folder=args.output_folder,
        )
