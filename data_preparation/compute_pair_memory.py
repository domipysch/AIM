"""
Compute estimated GPU memory for each pair in pairs.csv and write it back
as a new column `estimated_gpu_gb`.

Uses the same estimate_gpu_memory_gb function as main.py.
h5ad files are opened in backed='r' mode so the data matrix is never loaded.
K is taken as the maximum value across all model.K entries in experiment_config.yaml.

Usage
-----
    python compute_pair_memory.py \
        --pairs_csv  C:/Users/zi69hebi/Dev/10_Alignment/Data/01_Datasets/pairs.csv \
        --sc_dir     C:/Users/zi69hebi/Dev/10_Alignment/Data/01_Datasets/scRNA \
        --st_dir     C:/Users/zi69hebi/Dev/10_Alignment/Data/01_Datasets/ST \
        [--config    experiment_config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
import yaml

from main import estimate_gpu_memory_gb

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent / "experiment_config.yaml"


def _max_k_from_config(config_path: Path) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    k_val = cfg.get("model", {}).get("K", 1)
    if isinstance(k_val, list):
        return int(max(k_val))
    return int(k_val)


def main(pairs_csv: Path, sc_dir: Path, st_dir: Path, config_path: Path) -> None:
    pairs_csv = Path(pairs_csv)
    sc_dir = Path(sc_dir)
    st_dir = Path(st_dir)

    k = _max_k_from_config(config_path)
    logger.info("Using K=%d (max from config) for memory estimate.", k)

    pairs = pd.read_csv(pairs_csv)
    estimates: list[float | None] = []

    for _, row in pairs.iterrows():
        sc_path = sc_dir / f"{row['scName']}.h5ad"
        st_path = st_dir / f"{row['stName']}.h5ad"

        if not sc_path.exists() or not st_path.exists():
            logger.warning(
                "Pair %d: file(s) missing, skipping (sc=%s, st=%s).",
                row["PairID"],
                sc_path,
                st_path,
            )
            estimates.append(None)
            continue

        adata_sc = sc.read_h5ad(sc_path, backed="r")
        adata_st = sc.read_h5ad(st_path, backed="r")

        shared = len(set(adata_sc.var_names) & set(adata_st.var_names))
        gb = estimate_gpu_memory_gb(
            num_cells=adata_sc.n_obs,
            num_spots=adata_st.n_obs,
            g_sc=adata_sc.n_vars,
            g_st=adata_st.n_vars,
            num_genes_shared=shared,
            K=k,
        )

        adata_sc.file.close()
        adata_st.file.close()

        logger.info(
            "Pair %d (%s × %s): %d cells, %d spots, %d shared genes → %.2f GB",
            row["PairID"],
            row["scName"],
            row["stName"],
            adata_sc.n_obs,
            adata_st.n_obs,
            shared,
            gb,
        )
        estimates.append(round(gb, 2))

    pairs["estimated_gpu_gb"] = estimates
    pairs.to_csv(pairs_csv, index=False)
    logger.info("Written back to %s", pairs_csv)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Add estimated_gpu_gb column to pairs.csv."
    )
    parser.add_argument("--pairs_csv", type=Path, required=True)
    parser.add_argument("--sc_dir", type=Path, required=True)
    parser.add_argument("--st_dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="experiment_config.yaml (default: repo-root experiment_config.yaml)",
    )
    args = parser.parse_args()

    main(args.pairs_csv, args.sc_dir, args.st_dir, args.config)
