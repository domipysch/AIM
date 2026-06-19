# Generates noise-perturbed versions of an SC dataset for metric validation.
#
# For each noise level alpha, adds isotropic Gaussian noise scaled to the
# global expression standard deviation:
#   X_perturbed = clip(X + N(0, alpha * X.std()), min=0)
#
# alpha=0 → identity, alpha=1 → noise std equals the data std,
# higher alpha → progressively destroys cell-state structure.
# Values are clipped at 0 to preserve non-negativity.
#
# Usage:
#   python -m src.data_preparation.perturb_sc_noise \
#       --scdata <path/to/sc.h5ad> \
#       --noise_levels 0.5 1.0 2.0 5.0 \
#       [--output_dir <dir>]   # default: same directory as input
#       [--seed 42]

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def perturb_sc(
    sc_adata: ad.AnnData,
    noise_level: float,
    seed: int = 42,
) -> ad.AnnData:
    """
    Return a copy of sc_adata with Gaussian noise added to the expression matrix.
    Noise std = noise_level * global expression std. Values clipped at 0.
    """
    X = (
        sc_adata.X.toarray()
        if sp.issparse(sc_adata.X)
        else np.asarray(sc_adata.X, dtype=float)
    )

    if noise_level == 0.0:
        return sc_adata.copy()

    rng = np.random.default_rng(seed)
    sigma = noise_level * X.std()
    noise = rng.normal(0.0, sigma, size=X.shape)
    X_perturbed = np.clip(X + noise, a_min=0.0, a_max=None)

    result = sc_adata.copy()
    result.X = X_perturbed.astype(np.float32)
    result.uns["perturbation"] = {
        "type": "gaussian_noise",
        "noise_level": noise_level,
        "seed": seed,
    }
    return result


def main(sc_path: Path, noise_levels: list[float], output_dir: Path, seed: int) -> None:
    logger.info(f"Loading {sc_path} ...")
    sc_adata = ad.read_h5ad(sc_path)
    logger.info(f"  {sc_adata.n_obs} cells x {sc_adata.n_vars} genes")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sc_path.stem  # e.g. "sc"

    for alpha in noise_levels:
        perturbed = perturb_sc(sc_adata, alpha, seed=seed)
        label = f"{alpha:.2f}".replace(".", "p")  # e.g. 0.50 → "0p50"
        out_path = output_dir / f"{stem}_noise{label}.h5ad"
        perturbed.write_h5ad(out_path)
        logger.info(f"  Saved noise_level={alpha}: {out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate noise-perturbed SC datasets for metric validation."
    )
    parser.add_argument("--scdata", required=True, type=Path, help="Path to sc.h5ad")
    parser.add_argument(
        "--noise_levels",
        required=True,
        type=float,
        nargs="+",
        help="One or more noise level multipliers (e.g. 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=Path,
        help="Output directory (default: same directory as input)",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.scdata.parent

    main(args.scdata, args.noise_levels, output_dir, args.seed)
