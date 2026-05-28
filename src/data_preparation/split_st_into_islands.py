"""
Interactively split ST h5ad files into per-island files using drawn bounding boxes.

For each target file an interactive plot opens. Draw rectangles to define islands,
then press Enter. If any spots fall outside all boxes the plot reopens with those
spots highlighted in red so you can extend/add boxes. Once every spot is covered
the file is split and the database updated.

Controls:
  Click + drag      draw a rectangle
  a                 add the current selection as an island box
  u                 undo the last added box
  s                 skip this file (no split)
  Enter             confirm all boxes and proceed

Usage:
  python -m src.data_preparation.split_st_into_islands
      [--datasets 06 07 08]
      [--data-root <path>]
      [--dry-run]
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from sklearn.neighbors import NearestNeighbors

DEFAULT_DATA_ROOT = Path(r"C:\Users\zi69hebi\Dev\10_Alignment\Data\01_Datasets")
logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^(\d+)_(\d+)_(.+)$")

# Standard tab10 palette — avoids colourmap lookup across matplotlib versions
_TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


# ---------------------------------------------------------------------------
# Interactive bounding-box UI
# ---------------------------------------------------------------------------


class _BoxSelector:
    """One interactive session for a single ST file."""

    _COLORS = _TAB10

    def __init__(
        self,
        name: str,
        coords: np.ndarray,
        file_num: int,
        total_files: int,
        existing_boxes: list | None = None,
        uncovered_mask: np.ndarray | None = None,
    ):
        self.name = name
        self.coords = coords
        self.boxes: list[tuple] = list(existing_boxes) if existing_boxes else []
        self._pending: tuple | None = None
        self._skip = False

        fig, ax = plt.subplots(figsize=(11, 9))
        self.fig = fig
        self.ax = ax
        plt.subplots_adjust(top=0.87)

        # Spots
        if uncovered_mask is not None and uncovered_mask.any():
            covered = ~uncovered_mask
            ax.scatter(
                coords[covered, 0],
                coords[covered, 1],
                s=1,
                c="#aaaaaa",
                linewidths=0,
                zorder=1,
            )
            ax.scatter(
                coords[uncovered_mask, 0],
                coords[uncovered_mask, 1],
                s=3,
                c="red",
                linewidths=0,
                zorder=2,
                label=f"{int(uncovered_mask.sum()):,} uncovered (red)",
            )
            ax.legend(fontsize=9, loc="upper right")
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1], s=1, c="#aaaaaa", linewidths=0, zorder=1
            )

        # Pre-draw existing boxes
        self._patches: list[mpatches.Rectangle] = []
        for i, box in enumerate(self.boxes):
            self._patches.append(self._make_patch(box, i))
            ax.add_patch(self._patches[-1])

        suffix = (
            "  ← red = uncovered"
            if (uncovered_mask is not None and uncovered_mask.any())
            else ""
        )
        ax.set_title(
            f"[{file_num}/{total_files}]  {name}  ({len(coords):,} spots){suffix}\n"
            "drag → rectangle | a = add box | u = undo | s = skip file | Enter = done",
            fontsize=10,
        )
        ax.set_aspect("equal")

        self._label = ax.text(
            0.02,
            0.02,
            self._label_text(),
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        self._selector = RectangleSelector(
            ax,
            self._on_rect,  # type: ignore[arg-type]
            useblit=True,
            button=[1],  # type: ignore[arg-type, list-item]
            minspanx=1,
            minspany=1,
            spancoords="data",
        )
        fig.canvas.mpl_connect("key_press_event", self._on_key)  # type: ignore[arg-type]

    # --- helpers ---

    def _make_patch(self, box: tuple, index: int) -> mpatches.Rectangle:
        color = self._COLORS[index % len(self._COLORS)]
        x0, y0, x1, y1 = box
        return mpatches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
            label=f"Island {index + 1}",
        )

    def _label_text(self) -> str:
        return f"{len(self.boxes)} box(es) defined"

    def _refresh(self) -> None:
        self._label.set_text(self._label_text())
        handles = [self._make_patch(b, i) for i, b in enumerate(self.boxes)]
        if handles:
            self.ax.legend(handles=handles, fontsize=8, loc="upper right")
        else:
            legend = self.ax.get_legend()
            if legend:
                legend.remove()
        self.fig.canvas.draw_idle()

    # --- event handlers ---

    def _on_rect(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        self._pending = (
            min(eclick.xdata, erelease.xdata),
            min(eclick.ydata, erelease.ydata),
            max(eclick.xdata, erelease.xdata),
            max(eclick.ydata, erelease.ydata),
        )

    def _on_key(self, event) -> None:
        if event.key == "a" and self._pending is not None:
            patch = self._make_patch(self._pending, len(self.boxes))
            self.ax.add_patch(patch)
            self._patches.append(patch)
            self.boxes.append(self._pending)
            self._pending = None
            self._refresh()

        elif event.key == "u" and self.boxes:
            self.boxes.pop()
            self._patches[-1].remove()
            self._patches.pop()
            self._refresh()

        elif event.key == "s":
            self._skip = True
            plt.close(self.fig)

        elif event.key == "enter":
            plt.close(self.fig)

    # --- public ---

    def run(self) -> list | None:
        """Block until user finishes. Returns box list, or None if skipped."""
        plt.show(block=True)
        return None if self._skip else list(self.boxes)


# ---------------------------------------------------------------------------
# Coverage logic
# ---------------------------------------------------------------------------


def _assign(coords: np.ndarray, boxes: list) -> np.ndarray:
    """First-box-wins assignment. Unassigned spots get label -1."""
    labels = np.full(len(coords), -1, dtype=int)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        free = labels == -1
        in_box = (
            free
            & (coords[:, 0] >= x0)
            & (coords[:, 0] <= x1)
            & (coords[:, 1] >= y0)
            & (coords[:, 1] <= y1)
        )
        labels[in_box] = i
    return labels


def _assign_remaining_to_nearest(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fallback: assign still-uncovered spots to the nearest covered spot's label."""
    noise = labels == -1
    if not noise.any():
        return labels
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(coords[~noise])
    _, idx = nn.kneighbors(coords[noise])
    out = labels.copy()
    out[noise] = labels[~noise][idx[:, 0]]
    return out


def get_boxes_for_file(
    name: str,
    coords: np.ndarray,
    file_num: int,
    total_files: int,
) -> tuple[list, np.ndarray] | tuple[None, None]:
    """
    Run the interactive UI in a loop until every spot is covered.
    Returns (boxes, labels) or (None, None) if the user skipped the file.
    """
    existing_boxes: list | None = None
    uncovered_mask: np.ndarray | None = None

    while True:
        sel = _BoxSelector(
            name,
            coords,
            file_num,
            total_files,
            existing_boxes=existing_boxes,
            uncovered_mask=uncovered_mask,
        )
        boxes = sel.run()

        if boxes is None:
            return None, None  # type: ignore[return-value]  # user pressed 's'

        if not boxes:
            print(
                f"\n  No boxes defined — showing plot again. "
                "Press 's' to skip this file."
            )
            continue

        labels = _assign(coords, boxes)
        n_uncovered = int((labels == -1).sum())

        if n_uncovered == 0:
            return boxes, labels

        pct = 100 * n_uncovered / len(coords)
        print(
            f"\n  {n_uncovered:,} spots ({pct:.1f}%) are outside all boxes. "
            "Reopening — uncovered spots shown in red."
        )
        existing_boxes = boxes
        uncovered_mask = labels == -1


# ---------------------------------------------------------------------------
# Naming / sorting helpers
# ---------------------------------------------------------------------------


def _parse_name(name: str) -> tuple[str, str, str]:
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"ST name {name!r} does not match pattern nn_i_tissue")
    return m.group(1), m.group(2), m.group(3)


def _st_sort_key(row: dict) -> tuple[int, int, int]:
    name = row["Name"]
    m = re.match(r"^(\d+)_(\d+)_(\d+)_", name)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"^(\d+)_(\d+)_", name)
    if m:
        return (int(m.group(1)), int(m.group(2)), 0)
    m = re.match(r"^(\d+)_", name)
    if m:
        return (int(m.group(1)), 0, 0)
    return (999, 999, 999)


# ---------------------------------------------------------------------------
# Result plot
# ---------------------------------------------------------------------------


def _save_result_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    boxes: list,
    name: str,
    plot_dir: Path,
) -> None:
    n = len(boxes)
    colors = [_TAB10[i % len(_TAB10)] for i in range(n)]
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n):
        mask = labels == i
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=1,
            color=colors[i],
            linewidths=0,
            label=f"Island {i + 1}  ({int(mask.sum()):,} spots)",
        )
    ax.set_title(f"{name}  —  {n} islands  ({len(coords):,} spots)")
    ax.set_aspect("equal")
    ax.legend(markerscale=6, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    out = plot_dir / f"{name}_islands.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Plot saved: %s", out)


# ---------------------------------------------------------------------------
# Per-file split
# ---------------------------------------------------------------------------


def split_file(
    h5ad_path: Path,
    st_dir: Path,
    st_index: list,
    pairs: list,
    dry_run: bool,
    plot_dir: Path,
    file_num: int,
    total_files: int,
) -> bool:
    name = h5ad_path.stem
    logger.info("Loading %s  (%d / %d)", name, file_num, total_files)
    adata = ad.read_h5ad(h5ad_path)
    coords = adata.obsm["spatial"].astype(float)

    boxes, labels = get_boxes_for_file(name, coords, file_num, total_files)

    if boxes is None or labels is None:
        logger.info("  Skipped by user.")
        return False

    n_islands = len(boxes)
    island_counts = [int((labels == i).sum()) for i in range(n_islands)]
    logger.info("  %d islands: %s", n_islands, island_counts)

    if n_islands < 2:
        logger.info("  Only 1 box — no split needed for %s.", name)
        return False

    dataset_id, slice_idx, tissue = _parse_name(name)

    orig_idx_row = next((r for r in st_index if r["Name"] == name), None)
    if orig_idx_row is None:
        raise RuntimeError(f"{name} not found in ST/index.csv")
    orig_pairs = [p for p in pairs if p["stName"] == name]

    if dry_run:
        print(f"\n[DRY RUN] {name}: {n_islands} islands")
        for i in range(n_islands):
            new_name = f"{dataset_id}_{slice_idx}_{i + 1}_{tissue}"
            print(f"  island_{i + 1}: {island_counts[i]:,} spots  →  {new_name}")
        return True

    # Persist box definitions so the split is reproducible
    plot_dir.mkdir(parents=True, exist_ok=True)
    boxes_path = plot_dir / f"{name}_boxes.json"
    with open(boxes_path, "w") as f:
        json.dump({"name": name, "boxes": [list(b) for b in boxes]}, f, indent=2)

    _save_result_plot(coords, labels, boxes, name, plot_dir)

    # Update in-memory database
    st_index[:] = [r for r in st_index if r["Name"] != name]
    pairs[:] = [p for p in pairs if p["stName"] != name]

    total_spots = adata.n_obs
    for i in range(n_islands):
        island_num = i + 1
        new_name = f"{dataset_id}_{slice_idx}_{island_num}_{tissue}"
        mask = labels == i
        n_spots = island_counts[i]

        (adata[mask].copy()).write_h5ad(st_dir / f"{new_name}.h5ad")
        logger.info("  Wrote  %s  (%d spots)", new_name, n_spots)

        new_idx = {**orig_idx_row, "Name": new_name, "SpotCount": str(n_spots)}
        st_index.append(new_idx)

        frac = n_spots / total_spots
        for p in orig_pairs:
            pairs.append(
                {
                    **p,
                    "stName": new_name,
                    "estimated_gpu_gb": f"{float(p['estimated_gpu_gb']) * frac:.2f}",
                }
            )

    h5ad_path.unlink()
    logger.info("  Deleted original  %s", h5ad_path.name)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Interactively split ST files into islands using drawn bounding boxes."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["06", "07", "08"],
        help="Dataset prefixes to process (default: 06 07 08)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show UI and print planned splits without writing files",
    )
    args = parser.parse_args()

    st_dir = args.data_root / "ST"
    st_index_path = st_dir / "index.csv"
    pairs_path = args.data_root / "pairs.csv"
    plot_dir = st_dir / "island_split_plots"

    with open(st_index_path, newline="") as f:
        reader = csv.DictReader(f)
        st_fields: list[str] = list(reader.fieldnames or [])
        st_index = list(reader)

    with open(pairs_path, newline="") as f:
        reader = csv.DictReader(f)
        pairs_fields: list[str] = list(reader.fieldnames or [])
        pairs = list(reader)

    target_files = sorted(
        p for prefix in args.datasets for p in st_dir.glob(f"{prefix}_*.h5ad")
    )
    if not target_files:
        logger.error("No matching h5ad files found for prefixes: %s", args.datasets)
        sys.exit(1)

    logger.info("%d files to process.", len(target_files))

    n_split = 0
    for file_num, h5ad_path in enumerate(target_files, start=1):
        try:
            if split_file(
                h5ad_path,
                st_dir,
                st_index,
                pairs,
                args.dry_run,
                plot_dir,
                file_num,
                len(target_files),
            ):
                n_split += 1
        except Exception:
            logger.exception("Failed on %s", h5ad_path.name)
            sys.exit(1)

    if not args.dry_run and n_split > 0:
        for i, pair in enumerate(pairs):
            pair["PairID"] = str(i)

        with open(st_index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=st_fields)
            writer.writeheader()
            writer.writerows(sorted(st_index, key=_st_sort_key))  # type: ignore[arg-type]
        logger.info("Updated  %s", st_index_path)

        with open(pairs_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pairs_fields)
            writer.writeheader()
            writer.writerows(pairs)
        logger.info("Updated  %s", pairs_path)

    logger.info("Done. %d / %d files split.", n_split, len(target_files))


if __name__ == "__main__":
    main()
