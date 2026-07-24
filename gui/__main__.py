"""CLI launcher for the AIM GUI.

Parses the same up-front arguments the user supplies for a single pair
(``--scdata``/``--stdata``/``--output_dir`` + the K range), validates them, then
starts a Streamlit server running ``gui/app.py`` and forwards the arguments to it
after the ``--`` separator. Run from the repository root:

    python -m gui --scdata <sc.h5ad> --stdata <st.h5ad> --output_dir <out> \
        --k_min 2 --k_max 35 --k_step 1
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
    parser.add_argument("--scdata", type=Path, required=True, help="scRNA .h5ad path")
    parser.add_argument("--stdata", type=Path, required=True, help="ST .h5ad path")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output folder; each mapper's sweep is written to <output_dir>/<mapper>/",
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

    for label, path in (("--scdata", args.scdata), ("--stdata", args.stdata)):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")
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

    forwarded = [
        "--scdata",
        str(args.scdata),
        "--stdata",
        str(args.stdata),
        "--output_dir",
        str(args.output_dir),
        "--k_step",
        str(args.k_step),
    ]
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
