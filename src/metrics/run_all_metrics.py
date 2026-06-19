import argparse
from pathlib import Path
import logging
import sys
import anndata as ad
from anndata import AnnData
from .metrics_o2 import main as main2
from .metrics_o4 import main as main4
from .metrics_o2_permutation_test import main as main2permutation
from .metrics_o4_permutation_test import main as main4permutation

logger = logging.getLogger(__name__)


def main(
    sc_path: Path,
    st_path: Path,
    metrics: Path,
    result_gep: AnnData,
    name_suffix: str = "",
):
    """

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        result_gep: Predicted Z' (G x S)
        metrics:
        run_permutation_tests:
        name_suffix: Appended to output filenames before .json (e.g. "-det").

    Returns:

    """
    # Run metrics computations
    main2(sc_path, st_path, result_gep, metrics, name_suffix=name_suffix)
    # main4(sc_path, st_path, result_gep, metrics)

    # Run permutation tests
    # main2permutation(sc_path, st_path, result_gep, metrics)
    # main4permutation(sc_path, st_path, result_gep, metrics)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run all metrics on a result file")
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument("-r", "--result", type=Path, help="Path to result file")
    parser.add_argument(
        "-m", "--metrics", type=Path, help="Path to output metric folder"
    )

    args = parser.parse_args()

    logger.info("Starting metrics computation for:")
    logger.info("SC data path: %s", args.scdata)
    logger.info("ST data path: %s", args.stdata)
    logger.info("Result file path: %s", args.result)
    logger.info("Metrics output folder: %s", args.metrics)

    result_gep = ad.read_h5ad(args.result)
    main(args.scdata, args.stdata, args.metrics, result_gep)
