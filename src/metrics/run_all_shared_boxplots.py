from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import json
import sys
import argparse

logger = logging.getLogger(__name__)


def _extract_values(data) -> list:
    """Extract flat list of float values from both old (flat dict) and new (structured) cossim.json formats."""
    if isinstance(data, dict) and "values" in data:
        data = data["values"]
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    return []


def compute_medians(paths_to_jsons: list[Path], labels: list[str]) -> dict:
    """
    Compute the median of numeric values contained in each JSON file.
    """
    medians: dict[str, float | None] = {}
    for path, label in zip(paths_to_jsons, labels):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_values = _extract_values(data)
            nums = []
            for v in raw_values:
                if v is None:
                    continue
                try:
                    nums.append(float(v))
                except Exception:
                    continue

            if not nums:
                medians[label] = None
            else:
                medians[label] = float(np.median(nums))
        except Exception:
            medians[label] = None
    return medians


def create_shared_boxplot(
    paths_to_jsons: list[Path],
    labls: list[str],
    title: str,
    ylabel: str,
    output_path: Path | None = None,
):
    """
    Create a combined boxplot from multiple JSON files.
    Values are extracted from each file and plotted as side-by-side boxplots.
    If output_path is provided the figure is saved, otherwise it is shown interactively.
    """
    # Collect data from all JSON files
    all_data = []
    for path in paths_to_jsons:
        with open(path, "r") as f:
            data = json.load(f)
            all_data.append(_extract_values(data))

    # --- Modified/additional lines: increase font sizes ---
    # Increase global default font size
    plt.rcParams.update({"font.size": 14})
    # Optional: separate sizes for title and axis labels
    title_fontsize = 20
    label_fontsize = 18
    xtick_fontsize = 18
    # --------------------------------------------------------

    # Create boxplot
    n_boxes = len(paths_to_jsons)
    fig_width = max(6.0, n_boxes * 1.5)
    plt.figure(figsize=(fig_width, 6))
    plt.boxplot(all_data, tick_labels=labls)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(rotation=30, fontsize=xtick_fontsize)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()


def main(metrics_paths: list[Path], labels: list[str], output_folder: Path):

    # Create shared boxplot for o2, genewise
    create_shared_boxplot(
        [path / "o2" / "boxplots_per_gene" / "cossim.json" for path in metrics_paths],
        labels,
        "Metric for objective O2",
        "Genewise cosine similarity",
        output_path=output_folder / "o2_genewise_overall.pdf",
    )

    # Create shared boxplot for o2, spotwise
    create_shared_boxplot(
        [path / "o2" / "boxplots_per_spot" / "cossim.json" for path in metrics_paths],
        labels,
        "Metric for objective O2",
        "Spotwise cosine similarity",
        output_path=output_folder / "o2_spotwise_overall.pdf",
    )

    # Create shared boxplot for o4
    # create_shared_boxplot(
    #     [path / "o4" / "knn" / "cossim.json" for path in metrics_paths],
    #     labels,
    #     "Metric for objective O4",
    #     "Custom locality metric",
    #     output_path=output_folder / "o4_overall.pdf",
    # )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Create combined boxplots from multiple metric folders."
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

    logger.info("Create shared boxplots for:")
    logger.info("Metrics: %s", args.metrics)
    logger.info("Metrics output folder: %s", args.output_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if len(args.metrics) != len(args.labels):
        raise ValueError("Anzahl der --paths muss gleich der Anzahl der --labels sein.")

    main(args.metrics, args.labels, args.output_folder)
