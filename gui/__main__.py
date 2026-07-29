"""CLI launcher for the AIM GUI.

Starts a Streamlit server running ``gui/app.py``. All data arguments are
optional and only *prefill* the sidebar — the scRNA/ST paths, output directory,
and K step are set (and editable) in the app itself. Any supplied args are
forwarded after the ``--`` separator. Run from the repository root:

    python -m gui        # then set everything in the sidebar
    python -m gui --scdata <sc.h5ad> --stdata <st.h5ad> --output_dir <out>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gui",
        description="Launch the interactive AIM results GUI for a single sc/ST pair.",
    )
    parser.add_argument(
        "--scdata", type=Path, default=None, help="scRNA .h5ad path (prefills the UI)"
    )
    parser.add_argument(
        "--stdata", type=Path, default=None, help="ST .h5ad path (prefills the UI)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder (prefills the UI); each mapper's sweep is written to "
        "<output_dir>/<mapper>/",
    )
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--k_step", type=int, default=1)
    parser.add_argument(
        "--server_port",
        type=int,
        default=8501,
        help="Port for the Streamlit server (default 8501).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Paths are optional now (they can be set in the sidebar). Only validate the
    # ones that were supplied as prefill.
    for label, path in (("--scdata", args.scdata), ("--stdata", args.stdata)):
        if path is not None and not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "gui" / "app.py"
    src_path = repo_root / "src"

    env = os.environ.copy()
    # The app imports the in-repo packages (aim / analysis / plots / adata_schema)
    # which live under src/, exactly like every other entry point in this repo.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(src_path), env.get("PYTHONPATH", "")) if p
    )
    # Headless matplotlib: the GUI renders figures server-side and streams them.
    env["MPLBACKEND"] = "Agg"

    # Forward only the args that were supplied; the rest default in the UI.
    forwarded = ["--k_step", str(args.k_step)]
    if args.scdata is not None:
        forwarded += ["--scdata", str(args.scdata)]
    if args.stdata is not None:
        forwarded += ["--stdata", str(args.stdata)]
    if args.output_dir is not None:
        forwarded += ["--output_dir", str(args.output_dir)]
    if args.k_min is not None:
        forwarded += ["--k_min", str(args.k_min)]
    if args.k_max is not None:
        forwarded += ["--k_max", str(args.k_max)]

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.server_port),
        "--server.headless",
        "false",
        "--",
        *forwarded,
    ]
    print("Launching AIM GUI:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
