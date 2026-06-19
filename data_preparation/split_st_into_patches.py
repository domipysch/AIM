"""
Split a spatial transcriptomics dataset into an n×n grid of spatial patches.

Each spot is assigned to exactly one patch based on its coordinates in
adata_st.obsm["spatial"]. The coordinate space is divided into n equal-width
bins along each axis, yielding up to n×n patches.

The sc dataset is NOT touched — use the original sc.h5ad with every patch.

Usage:
  python -m src.data_preparation.split_st_into_patches -d <st_path> -n <grid_size>

  # Custom output directory:
  python -m src.data_preparation.split_st_into_patches -d <st_path> -n 3 -o <output_dir>

Output structure:
  <output_dir>/
    patch_0_0/st.h5ad
    patch_0_1/st.h5ad
    ...
    patch_{n-1}_{n-1}/st.h5ad
"""

import argparse
import logging
import sys
from pathlib import Path

import csv
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)


def split_st_into_patches(st_path: Path, n: int, output_dir: Path) -> None:
    """
    Load the given st.h5ad, split into an n×n spatial grid,
    and write each patch as st.h5ad into a subdirectory of output_dir.

    Every spot is guaranteed to appear in exactly one patch.
    Empty patches (no spots) are skipped with a warning.

    Args:
        st_path: Full path to the ST h5ad file.
        n: Grid dimension. Produces up to n×n patches.
        output_dir: Root directory where patch subdirectories are written.
    """
    if not st_path.exists():
        raise FileNotFoundError(f"ST file not found: {st_path}")

    logger.info("Loading ST data from %s", st_path)
    adata_st = ad.read_h5ad(st_path)

    if "spatial" not in adata_st.obsm:
        raise ValueError(
            "adata_st.obsm['spatial'] is missing — cannot split spatially. "
            "Run convert_csv_to_h5ad first if working with legacy CSV data."
        )

    coords = adata_st.obsm["spatial"]  # S x 2
    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)

    logger.info(
        "Loaded %d spots. Coordinate ranges: x=[%.4f, %.4f], y=[%.4f, %.4f]",
        adata_st.n_obs,
        x.min(),
        x.max(),
        y.min(),
        y.max(),
    )

    # --- Bin assignment ---
    # Build n+1 evenly spaced edges covering the full coordinate range.
    # np.digitize returns 1-indexed bins; subtract 1 for 0-indexed.
    # Spots exactly at the maximum edge get bin index n (out of [0, n-1]),
    # so clip them to n-1. This puts boundary spots into the last bin and
    # ensures every spot belongs to exactly one patch.
    x_edges = np.linspace(x.min(), x.max(), n + 1)
    y_edges = np.linspace(y.min(), y.max(), n + 1)

    x_bins = np.clip(
        np.digitize(x, x_edges) - 1, 0, n - 1
    )  # shape (S,), values in [0, n-1]
    y_bins = np.clip(
        np.digitize(y, y_edges) - 1, 0, n - 1
    )  # shape (S,), values in [0, n-1]

    output_dir.mkdir(parents=True, exist_ok=True)

    total_spots_written = 0
    patches_written = 0
    summary_rows = []

    for row in range(n):
        for col in range(n):
            mask = (x_bins == row) & (y_bins == col)
            n_spots = int(mask.sum())

            if n_spots == 0:
                logger.warning("Patch (%d, %d) contains no spots — skipping.", row, col)
                summary_rows.append({"patch": f"patch_{row}_{col}", "spots": 0})
                continue

            patch_adata = adata_st[mask].copy()
            patch_dir = output_dir / f"patch_{row}_{col}"
            patch_dir.mkdir(parents=True, exist_ok=True)
            out_path = patch_dir / "st.h5ad"
            patch_adata.write_h5ad(out_path)

            logger.info("Patch (%d, %d): %5d spots -> %s", row, col, n_spots, out_path)

            total_spots_written += n_spots
            patches_written += 1
            summary_rows.append({"patch": f"patch_{row}_{col}", "spots": n_spots})

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch", "spots"])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Summary written to %s", summary_path)

    # Sanity check: every spot must appear in exactly one patch
    assert total_spots_written == adata_st.n_obs, (
        f"Spot count mismatch after splitting: wrote {total_spots_written}, "
        f"expected {adata_st.n_obs}. This is a bug."
    )

    logger.info(
        "Done. %d / %d patches written, %d total spots.",
        patches_written,
        n * n,
        total_spots_written,
    )
    print(f"Patch st.h5ad files:  {output_dir}/patch_<row>_<col>/st.h5ad")

    # --- Visualisation ---
    # Assign each spot a flat patch index (row * n + col) for colouring.
    patch_index = x_bins * n + y_bins  # shape (S,), values in [0, n*n - 1]
    n_patches = n * n

    cmap = plt.get_cmap("tab20" if n_patches <= 20 else "turbo", n_patches)
    colors = [cmap(i) for i in range(n_patches)]

    fig, ax = plt.subplots(figsize=(7, 7))
    for idx in range(n_patches):
        mask = patch_index == idx
        if not mask.any():
            continue
        row, col = divmod(idx, n)
        ax.scatter(
            x[mask],
            y[mask],
            s=1,
            color=colors[idx],
            label=f"patch_{row}_{col}",
            linewidths=0,
        )

    ax.set_title(
        f"ST spots coloured by {n}×{n} patch assignment ({adata_st.n_obs} spots)"
    )
    ax.set_xlabel("x (spatial coord 0)")
    ax.set_ylabel("y (spatial coord 1)")
    ax.legend(
        markerscale=6,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
    )
    ax.set_aspect("equal")

    plot_path = output_dir / "patch_overview.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Patch overview plot saved to %s", plot_path)


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Split an ST dataset into an n×n grid of spatial patches. "
            "Each spot is assigned to exactly one patch. "
            "Use the original sc.h5ad together with each patch's st.h5ad."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        required=True,
        help="Full path to the ST h5ad file (e.g. .../ST/01_mouse-ssp.h5ad)",
    )
    parser.add_argument(
        "-n",
        "--grid",
        type=int,
        required=True,
        help="Grid size n: splits the coordinate space into n×n patches (e.g. 3 → up to 9 patches)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        help=(
            "Output directory for patch subfolders "
            "(default: <dataset_folder>_patches_<n>x<n> next to the dataset)"
        ),
    )
    args = parser.parse_args()

    if args.grid < 1:
        print("ERROR: -n must be >= 1", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        args.output
        if args.output is not None
        else args.dataset.parent
        / f"{args.dataset.stem}_patches_{args.grid}x{args.grid}"
    )

    split_st_into_patches(args.dataset, args.grid, output_dir)


if __name__ == "__main__":
    main()
