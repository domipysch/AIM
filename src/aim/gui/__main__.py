"""Launcher for the AIM GUI.

Starts a Streamlit server running ``aim/gui/app.py``. The only argument is the
server port; everything else - the scRNA/ST paths, output directory, K step and
the agglomeration linkage - is set in the app's sidebar.

    aim gui                      # then set everything in the sidebar
    aim gui --server_port 8600
    python -m aim.gui            # equivalent
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.resources import as_file, files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aim gui",
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


def launch(server_port: int = 8501) -> int:
    """Run ``streamlit run aim/gui/app.py`` on ``server_port``; return its exit code.

    The app imports the installed ``aim`` package directly, so PYTHONPATH does
    not need to be set.
    """
    app_resource = files("aim.gui") / "app.py"
    with as_file(app_resource) as app_path:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(server_port),
            "--server.headless",
            "false",
        ]
        print("Launching AIM GUI:", " ".join(cmd), flush=True)
        return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return launch(server_port=args.server_port)


if __name__ == "__main__":
    raise SystemExit(main())
