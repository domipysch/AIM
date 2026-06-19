import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


plt.rcParams.update({"font.size": 14})


def combined_boxplot(json_paths, labels, ylabel, title, output: Path):
    """
    Create one boxplot from multiple permutation test JSON files.
    (e.g. per method: all 6 configs)
    Args:
        json_paths:
        output:
        labels:
        title:

    Returns:

    """
    json_paths = [Path(p) for p in json_paths]
    boxes = []
    originals = []
    names = []

    for i, p in enumerate(json_paths):
        if not p.exists():
            print(f"Datei nicht gefunden, übersprungen: {p}", file=sys.stderr)
            continue
        try:
            d = json.loads(Path(p).read_text())
            perm, orig = list(map(float, d["T_permuted"])), float(d["T_original"])
        except Exception as e:
            print(f"Fehler beim Lesen {p}, übersprungen: {e}", file=sys.stderr)
            continue
        boxes.append(list(map(float, perm)))
        originals.append(float(orig))
        if labels and i < len(labels):
            names.append(labels[i])
        else:
            names.append(p.stem)

    if not boxes:
        print("Keine gültigen JSON-Dateien gefunden.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(boxes)), 5))
    ax.boxplot(boxes, patch_artist=True, medianprops={"color": "black"})
    xs = list(range(1, len(boxes) + 1))
    ax.scatter(xs, originals, color="red", marker="D", zorder=5)
    ax.set_xticks(xs)
    ax.set_title(title)
    # explizite Fontgrößen für xticks und ylabel
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)


def main(metrics_paths, labels, output_folder):

    # Create shared boxplot for permutation test o2
    combined_boxplot(
        [
            path / "o2" / "permutation_test" / "permutation_test_per_gene.json"
            for path in metrics_paths
        ],
        labels,
        "T = Sum of cos sim",
        "o2 permutation test across runs",
        output_folder / "o2_permutation.pdf",
    )

    # Create shared boxplot for permutation test o4
    # combined_boxplot(
    #     [path / "o4" / "knn" / "permutation_test.json" for path in metrics_paths],
    #     labels,
    #     "T = Sum of custom locality metric",
    #     "o4 permutation test across runs",
    #     output_folder / "o4_permutation.pdf",
    # )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Create boxplots for permutation test results (single JSON per call)"
    )
    parser.add_argument(
        "-m", "--metrics", nargs="+", type=Path, help="Path to output metric folders"
    )
    parser.add_argument(
        "-l", "--labels", nargs="+", type=str, help="Label for each box"
    )
    parser.add_argument(
        "-o", "--output_folder", type=Path, help="Path to output folder"
    )
    args = parser.parse_args()

    logger.info("Create shared permutation boxplots for:")
    logger.info("Metrics: %s", args.metrics)
    logger.info("Metrics output folder: %s", args.output_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if len(args.metrics) != len(args.labels):
        raise ValueError("Anzahl der --paths muss gleich der Anzahl der --labels sein.")

    main(args.metrics, args.labels, args.output_folder)
