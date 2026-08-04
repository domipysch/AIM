"""CLI launcher for the AIM GUI.

Starts a Streamlit server running ``gui/app.py``. The only argument is the
server port; everything else — the scRNA/ST paths, output directory, K step and
the agglomeration linkage — is set in the app's sidebar. Run from the repository
root:

    python -m gui                     # then set everything in the sidebar
    python -m gui --server_port 8600
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
        description="Launch the interactive AIM results GUI (configure everything "
        "in the sidebar).",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8501,
        help="Port for the Streamlit server (default 8501).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "gui" / "app.py"
    src_path = repo_root / "src"

    env = os.environ.copy()
    # The app imports the in-repo packages (aim / analysis / plots / adata_schema)
    # which live under src/, exactly like every other entry point in this repo.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(src_path), env.get("PYTHONPATH", "")) if p
    )

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
    ]
    print("Launching AIM GUI:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
