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
) -> tuple[AnnData, AnnData]:
    """
    Run DOT alignment by calling run_dot.R via Rscript.
    Saves gep_prob.h5ad, gep_det.h5ad, mapping_prob.h5ad, mapping_det.h5ad
    to output_folder and returns (gep_prob, gep_det).

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        cell_type_key: obs column to use as annotation; pass "cellID" to map individual cells.
        output_folder: Folder where all outputs are written.

    Returns:
        Tuple (gep_prob, gep_det), each AnnData with obs=genes, var=spots (G x S layout).
    """
    annotation_key = cell_type_key if cell_type_key else "cellID"

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    gep_prob_csv = output_folder / "gep_prob.csv"
    gep_det_csv = output_folder / "gep_det.csv"
    map_prob_csv = output_folder / "mapping_prob.csv"
    map_det_csv = output_folder / "mapping_det.csv"

    cmd = [
        _find_rscript(),
        R_SCRIPT,
        str(sc_path),
        str(st_path),
        "HSO",
        annotation_key,
        str(gep_prob_csv),
        str(gep_det_csv),
        str(map_prob_csv),
        str(map_det_csv),
    ]
    logger.info("Running DOT via R: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    def _csv_to_gep_h5ad(csv_path: Path, h5ad_path: Path) -> AnnData:
        df = pd.read_csv(csv_path, header=0, index_col=0)
        adata = AnnData(
            X=np.asarray(df.values, dtype=np.float32),
            obs=pd.DataFrame(index=df.index),
            var=pd.DataFrame(index=df.columns),
        )
        adata.write_h5ad(h5ad_path)
        csv_path.unlink()
        return adata

    def _csv_to_mapping_h5ad(csv_path: Path, h5ad_path: Path) -> AnnData:
        # R writes weights as S×T (rows=spots, cols=cell_types)
        # Transpose to T×S to match Tangram/TACCO layout (obs=cell_types, var=spots)
        df = pd.read_csv(csv_path, header=0, index_col=0)
        adata = AnnData(
            X=np.asarray(df.values.T, dtype=np.float32),
            obs=pd.DataFrame(index=df.columns),
            var=pd.DataFrame(index=df.index),
        )
        adata.write_h5ad(h5ad_path)
        csv_path.unlink()
        return adata

    gep_prob = _csv_to_gep_h5ad(gep_prob_csv, output_folder / "gep_prob.h5ad")
    gep_det = _csv_to_gep_h5ad(gep_det_csv, output_folder / "gep_det.h5ad")
    _csv_to_mapping_h5ad(map_prob_csv, output_folder / "mapping_prob.h5ad")
    _csv_to_mapping_h5ad(map_det_csv, output_folder / "mapping_det.h5ad")

    logger.info("All DOT outputs written to %s", output_folder)
    return gep_prob, gep_det


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
        "--cell_type_key",
        type=str,
        required=False,
        default="cellType",
        help="What cell type key to load from sc data as cell type annotation to be mapped. If not provided, no cell type annotation is loaded and mapping is done cell-based.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=Path,
        required=True,
        help="Folder where to store results",
    )
    args = parser.parse_args()

    dot_align_data(
        args.scdata,
        args.stdata,
        cell_type_key=args.cell_type_key,
        output_folder=args.output_folder,
    )
