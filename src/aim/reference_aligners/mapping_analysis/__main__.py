"""CLI: analyze a reference aligner's mapping_prob.h5ad output.

    python -m reference_aligners.mapping_analysis \\
        --scdata <sc.h5ad> --stdata <st.h5ad> \\
        --mapping_folder <folder containing mapping_prob.h5ad> \\
        --cell_type_key cellType

Works identically for Tangram, TACCO, and DOT outputs.
"""

import argparse
import logging
import sys
from pathlib import Path

# analyze.py imports `metrics.*` / `utils`, which live under src/ — add it to
# sys.path here so this CLI works regardless of whether the caller remembers
# to set PYTHONPATH=src.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from aim.reference_aligners.mapping_analysis.analyze import analyze_mapping

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Analyze a reference aligner's mapping_prob.h5ad output "
        "(Tangram/TACCO/DOT share the same format, so one script covers all three)."
    )
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "--mapping_folder",
        type=Path,
        required=True,
        help="Folder containing mapping_prob.h5ad (as written by run_tangram/"
        "run_tacco/run_dot); analysis/ is written here",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        default="cellType",
        help="obs column in sc data with the same values as the mapping's var_names",
    )
    args = parser.parse_args()

    analyze_mapping(
        args.scdata,
        args.stdata,
        args.mapping_folder,
        cell_type_key=args.cell_type_key,
    )
