"""
CLI Runner for AIM.

Parses arguments into an ``aim.AIMConfig`` and runs the sweep for a single sc/st
pair or for every row of a pairs.csv. The method itself lives in the ``aim``
package (see ``src/aim/__init__.py`` for the overview and module map); this file
holds only the CLI and the thin single-pair / batch drivers.

Run (single pair):
    PYTHONPATH=src python main.py \
        --scdata <sc.h5ad> --stdata <st.h5ad> --output_dir <out> \
        [--mapping greedy|learned] \
        [--leiden_resolution 3.0] [--normalize_and_log] \
        [--k_min 1] [--k_max <L>] [--k_step 1] \
        # learned-mode only:
        [--epochs 400] [--lr 0.02] [--lambda_spot_gini 1.0] \
        [--spot_gini_warmup_frac 0.5]

Run (all pairs in pairs.csv, one after the other):
    PYTHONPATH=src python main.py \
        --pairs_csv <pairs.csv> --sc_dir <scRNA dir> --st_dir <ST dir> \
        --output_dir <out> [same hyperparameter flags as above]
"""

import argparse
import csv
import logging
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

# conda ships both Intel (libiomp) and LLVM (libomp) OpenMP runtimes; threadpoolctl
# warns about the duplicate load on every sklearn/scanpy call. It is harmless for
# our numeric usage, so silence that one RuntimeWarning before the heavy imports.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*Found Intel OpenMP.*LLVM OpenMP",
    category=RuntimeWarning,
)

# docrep (pulled in by squidpy) emits a SyntaxWarning for every unrecognised
# docstring key (e.g. 'n_jobs', 'show_progress_bar'); harmless doc-parsing noise.
warnings.filterwarnings(
    "ignore",
    message=r".*is not a valid key!",
    category=SyntaxWarning,
)

import pandas as pd
import yaml

from aim import run
from aim.aim_config import MAPPING_CHOICES, AIMConfig

logger = logging.getLogger(__name__)


def run_one_pair(
    sc_path: Path, st_path: Path, output_folder: Path, config: AIMConfig
) -> pd.DataFrame:
    """Write config.yaml + run the K-sweep for a single sc/st pair."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "config.yaml", "w") as f:
        yaml.safe_dump({**asdict(config)}, f, sort_keys=False)

    return run(
        sc_path=sc_path,
        st_path=st_path,
        output_folder=output_folder,
        mapper=config.build_mapper(),
        generate_pdf=True,
        leiden_resolution=config.leiden_resolution,
        k_min=config.k_min,
        k_max=config.k_max,
        k_step=config.k_step,
    )


def run_batch(
    pairs_csv: Path,
    sc_dir: Path,
    st_dir: Path,
    output_dir: Path,
    config: AIMConfig,
) -> list[str]:
    """Run the sweep for every row of pairs.csv. Returns the list of error messages
    (empty if all pairs succeeded)."""
    with open(pairs_csv, newline="") as fh:
        all_pairs = list(csv.DictReader(fh))
    logger.info("Loaded %d pairs from %s", len(all_pairs), pairs_csv)

    errors: list[str] = []
    for pair in all_pairs:
        pair_id = int(pair["PairID"])
        sc_name = pair["scName"]
        st_name = pair["stName"]
        sc_path = sc_dir / f"{sc_name}.h5ad"
        st_path = st_dir / f"{st_name}.h5ad"

        missing = [p for p in (sc_path, st_path) if not p.exists()]
        if missing:
            msg = f"[Pair {pair_id:>3}] Missing files: {[str(p) for p in missing]}"
            logger.error(msg)
            errors.append(msg)
            continue

        pair_output = output_dir / f"{pair_id:03d}_{sc_name}__{st_name}"
        logger.info("[Pair %3d] %s x %s -> %s", pair_id, sc_name, st_name, pair_output)
        try:
            run_one_pair(sc_path, st_path, pair_output, config)
        except Exception as exc:
            msg = f"[Pair {pair_id:>3}] FAILED: {exc}"
            logger.error(msg)
            errors.append(msg)

    if errors:
        logger.warning("%d pair(s) failed:", len(errors))
        for e in errors:
            logger.warning("  %s", e)
    else:
        logger.info("All pairs completed successfully.")
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agglomerative-K sweep — modular greedy / learned spot-to-state mapping"
    )
    # Single-pair mode
    parser.add_argument(
        "--scdata", type=Path, default=None, help="Single-pair mode: sc .h5ad path"
    )
    parser.add_argument(
        "--stdata", type=Path, default=None, help="Single-pair mode: ST .h5ad path"
    )
    # Batch mode
    parser.add_argument(
        "--pairs_csv", type=Path, default=None, help="Batch mode: path to pairs.csv"
    )
    parser.add_argument(
        "--sc_dir",
        type=Path,
        default=None,
        help="Batch mode: folder containing scRNA .h5ad files",
    )
    parser.add_argument(
        "--st_dir",
        type=Path,
        default=None,
        help="Batch mode: folder containing ST .h5ad files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder (single pair) or root for all pair outputs (batch)",
    )
    # Method knobs
    parser.add_argument(
        "--mapping",
        choices=list(MAPPING_CHOICES),
        default="greedy",
        help="Spot-to-state mapping: 'greedy' (zero-parameter nearest-centroid, "
        "default), 'learned' (gradient-descent soft P), or an external reference "
        "aligner 'tangram' / 'tacco' / 'dot' (each runs out-of-process in its own "
        "conda env, once per K). The learned-mode flags below apply only to 'learned'.",
    )
    parser.add_argument("--leiden_resolution", type=float, default=3.0)
    parser.add_argument("--normalize_and_log", action="store_true", default=False)
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--k_step", type=int, default=1)
    # Learned-mode only
    parser.add_argument(
        "--epochs", type=int, default=400, help="[learned] per-K deconvolution epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.02, help="[learned] learning rate"
    )
    parser.add_argument(
        "--lambda_spot_gini",
        type=float,
        default=1.0,
        help="[learned] quadratic (Gini/Tsallis-2) spot sharpener on P — the strong "
        "one-hot lever. Set 0 to disable.",
    )
    parser.add_argument(
        "--spot_gini_warmup_frac",
        type=float,
        default=0.5,
        help="[learned] fraction of epochs to train with spot_gini OFF before "
        "ramping it in linearly (e.g. 0.5 = pure reconstruction for the first half, "
        "then ramp to full by the last epoch). 0 = constant weight from the start.",
    )
    parser.add_argument("--logging", choices=["normal", "verbose"], default="normal")
    return parser


def _config_from_args(args: argparse.Namespace) -> AIMConfig:
    return AIMConfig(
        mapping=args.mapping,
        leiden_resolution=args.leiden_resolution,
        normalize_and_log=args.normalize_and_log,
        epochs=args.epochs,
        lr=args.lr,
        lambda_spot_gini=args.lambda_spot_gini,
        spot_gini_warmup_frac=args.spot_gini_warmup_frac,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.logging == "verbose" else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = _config_from_args(args)

    batch_mode = args.pairs_csv is not None
    if batch_mode:
        if args.sc_dir is None or args.st_dir is None or args.output_dir is None:
            parser.error("--pairs_csv requires --sc_dir, --st_dir, and --output_dir")
        run_batch(args.pairs_csv, args.sc_dir, args.st_dir, args.output_dir, config)
    else:
        if args.scdata is None or args.stdata is None or args.output_dir is None:
            parser.error(
                "Provide either --scdata/--stdata (single pair) "
                "or --pairs_csv/--sc_dir/--st_dir (batch)"
            )
        run_one_pair(args.scdata, args.stdata, args.output_dir, config)


if __name__ == "__main__":
    main()
