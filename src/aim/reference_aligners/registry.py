"""Registry of the external reference aligners and the runner that executes them.

Adding a reference aligner is two steps:

  1. Write ``aim/reference_aligners/run_<name>.py`` satisfying the CLI contract
     (see ``ReferenceAligner`` / ``run_aligner`` below, or the README section
     "Adding a reference aligner").
  2. Add one ``ReferenceAligner(...)`` line to ``REFERENCE_ALIGNERS``.

The aligner is then available in ``aim run --mapping <name>``,
``aim run-reference --aligner <name>``, and as an AIM reference mapper
(``ReferenceMapper``).

Import-light (stdlib only) so it is safe to import from any conda env, including
``aim_env`` where torch/scanpy/aligner deps are not installed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

logger = logging.getLogger(__name__)

# The file every wrapper must write, and every caller reads back.
MAPPING_PROB_FILENAME = "mapping_prob.h5ad"


@dataclass(frozen=True)
class ReferenceAligner:
    """One external aligner: the conda env it runs in and its ``python -m`` module.

    The wrapper module MUST implement this CLI contract:

    * **args:** ``--scdata``, ``--stdata``, ``--output_folder``, ``--cell_type_key``.
    * **input:** ``scdata`` carries the states/types in ``obs[cell_type_key]``;
      ``stdata`` shares genes with ``scdata`` and carries spatial coordinates in
      ``obsm['spatial']``.
    * **output:** it writes ``<output_folder>/mapping_prob.h5ad`` — an AnnData with
      ``obs`` = spots (in ST order), ``var`` = state/type names, and ``X`` = a
      float32 soft assignment matrix (S x T).

    Invoked with only the four standard args, the wrapper must reproduce its
    canonical baseline mapping (bake any other choices into the CLI defaults) so
    that every caller runs it the same way.
    """

    name: str
    conda_env: str
    module: str


REFERENCE_ALIGNERS: dict[str, ReferenceAligner] = {
    a.name: a
    for a in (
        ReferenceAligner(
            "tangram", "tangram_env", "aim.reference_aligners.run_tangram"
        ),
        ReferenceAligner("tacco", "tacco_env", "aim.reference_aligners.run_tacco"),
        ReferenceAligner("dot", "dot_env", "aim.reference_aligners.run_dot"),
    )
}


def conda_exe() -> str:
    """Locate a conda launcher usable from subprocess. ``CONDA_EXE`` (set by conda
    activation) points to a real executable; ``shutil.which`` is the fallback (on
    Windows plain "conda" is a .bat that subprocess can't resolve without help)."""
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("mamba")
    if not exe:
        raise RuntimeError(
            "conda not found: reference aligners run in their own env via "
            "`conda run`, which needs conda on PATH (or CONDA_EXE set)."
        )
    return exe


def run_aligner(
    name: str,
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
    cell_type_key: str,
) -> Path:
    """Run reference aligner ``name`` in its own conda env via ``conda run``.

    Executes ``python -m <module> --scdata … --stdata … --output_folder …
    --cell_type_key …`` against the aligner's registered env and returns the path
    to the ``mapping_prob.h5ad`` it must produce. Raises ``RuntimeError`` (with the
    stderr tail) if the aligner exits non-zero or leaves no mapping behind.

    Both ``ReferenceMapper`` (per K, inside an AIM sweep) and the ``run-reference``
    driver call this, so it is the only path that executes an aligner.
    """
    if name not in REFERENCE_ALIGNERS:
        raise ValueError(
            f"unknown reference aligner {name!r}; known: {tuple(REFERENCE_ALIGNERS)}"
        )
    aligner = REFERENCE_ALIGNERS[name]
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cmd = [
        conda_exe(),
        "run",
        "-n",
        aligner.conda_env,
        "python",
        "-m",
        aligner.module,
        "--scdata",
        str(sc_path),
        "--stdata",
        str(st_path),
        "--output_folder",
        str(output_folder),
        "--cell_type_key",
        cell_type_key,
    ]
    logger.info(
        "run_aligner[%s] -> conda run -n %s python -m %s",
        name,
        aligner.conda_env,
        aligner.module,
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-2000:]
        raise RuntimeError(
            f"{name} aligner failed (exit {exc.returncode}).\n"
            f"--- stderr tail ---\n{tail}"
        ) from exc

    out_path = output_folder / MAPPING_PROB_FILENAME
    if not out_path.exists():
        raise RuntimeError(f"{name} produced no mapping at {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Persistent worker: read the sc/st pair once, map many K over its lifetime.
#
# ``run_aligner`` above pays a fresh ``conda run`` cold start (env activation +
# torch/aligner imports) *and* re-reads the sc/st h5ad on every call. Inside an
# AIM sweep that repeats per K. ``AlignerWorker`` keeps one aligner process
# alive for the whole K-loop: it loads the pair once, then serves per-K jobs
# through a small file-based queue (robust to the aligner libraries printing to
# stdout/stderr, and to ``conda run`` buffering, which a stdin/stdout protocol is
# not). Each wrapper opts in by supporting ``--server`` (see ``serve_loop``).
# The all-pairs batch driver, which runs one alignment per pair, keeps using the
# simpler one-shot ``run_aligner``.
# ---------------------------------------------------------------------------

# Poll interval (s) for the file-based job queue. Each K job is an aligner run
# (seconds to minutes), so a coarse poll adds negligible latency.
_WORKER_POLL_SECONDS = 0.05


def serve_loop(
    control_dir: str | Path, handle_job: Callable[[str, Path], None]
) -> None:
    """Worker side of :class:`AlignerWorker`: process jobs until told to stop.

    Runs inside the aligner's conda env (called from a wrapper's ``--server``
    branch). Polls ``control_dir`` for ``job_{n}.json`` files written by the
    parent, calls ``handle_job(cell_type_key, output_folder)`` for each (which
    must write ``mapping_prob.h5ad`` or raise), and reports the outcome back via
    ``job_{n}.done``. Returns when the parent drops a ``stop`` sentinel.

    Kept stdlib-only so it imports cleanly in every env.
    """
    control_dir = Path(control_dir)
    n = 0
    while True:
        if (control_dir / "stop").exists():
            return
        job_path = control_dir / f"job_{n:06d}.json"
        if not job_path.exists():
            time.sleep(_WORKER_POLL_SECONDS)
            continue
        spec = json.loads(job_path.read_text())
        payload: dict[str, object]
        try:
            handle_job(spec["cell_type_key"], Path(spec["output_folder"]))
            payload = {"ok": True}
        except Exception as exc:  # noqa: BLE001 — reported to the parent, not swallowed
            payload = {"ok": False, "error": repr(exc)}
        done_path = control_dir / f"job_{n:06d}.done"
        tmp = done_path.with_suffix(".done.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(done_path)  # atomic: parent never reads a half-written ack
        n += 1


class AlignerWorker:
    """Long-lived aligner subprocess that maps many K from one loaded sc/st pair.

    Use as a context manager; call :meth:`run_job` once per K::

        with AlignerWorker("tangram", sc_path, st_path) as worker:
            for k in ks:
                mapping = worker.run_job(state_key(k), out_dir(k))

    Jobs are submitted strictly one at a time (submit job ``n``, wait for its
    ``.done``), so the parent's and worker's counters stay in lock-step.
    """

    def __init__(self, name: str, sc_path: str | Path, st_path: str | Path) -> None:
        if name not in REFERENCE_ALIGNERS:
            raise ValueError(
                f"unknown reference aligner {name!r}; known: {tuple(REFERENCE_ALIGNERS)}"
            )
        self.name = name
        self._aligner = REFERENCE_ALIGNERS[name]
        self._sc_path = Path(sc_path)
        self._st_path = Path(st_path)
        self._counter = 0
        self._proc: subprocess.Popen | None = None
        self._log: TextIO | None = None

    def __enter__(self) -> "AlignerWorker":
        return self.start()

    def start(self) -> "AlignerWorker":
        """Launch the worker subprocess. Idempotent-safe to pair with :meth:`stop`."""
        self._control_tmp = tempfile.TemporaryDirectory(
            prefix=f"aim_worker_{self.name}_"
        )
        self._control = Path(self._control_tmp.name)
        self._log_path = self._control / "worker.log"
        self._log = open(self._log_path, "w")
        cmd = [
            conda_exe(),
            "run",
            "--no-capture-output",  # let the child's stdio flow to our log unbuffered
            "-n",
            self._aligner.conda_env,
            "python",
            "-m",
            self._aligner.module,
            "--server",
            "--scdata",
            str(self._sc_path),
            "--stdata",
            str(self._st_path),
            "--control_dir",
            str(self._control),
        ]
        logger.info(
            "AlignerWorker[%s] starting -> conda run -n %s python -m %s --server",
            self.name,
            self._aligner.conda_env,
            self._aligner.module,
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return self

    def run_job(self, cell_type_key: str, output_folder: str | Path) -> Path:
        """Map one K: hand the worker a cell-type key + output folder and block
        until it writes ``mapping_prob.h5ad``. Returns that path; raises with the
        worker-log tail if the job fails or the worker dies."""
        if self._proc is None:
            raise RuntimeError("AlignerWorker must be used as a context manager")
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        n = self._counter
        self._counter += 1
        job_path = self._control / f"job_{n:06d}.json"
        done_path = self._control / f"job_{n:06d}.done"
        tmp = job_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"cell_type_key": cell_type_key, "output_folder": str(output_folder)}
            )
        )
        tmp.replace(job_path)  # atomic: worker never reads a half-written job

        while not done_path.exists():
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} worker exited (code {self._proc.returncode}) "
                    f"before finishing K-job {n}.\n"
                    f"--- worker.log tail ---\n{self._log_tail()}"
                )
            time.sleep(_WORKER_POLL_SECONDS)

        result = json.loads(done_path.read_text())
        if not result.get("ok"):
            raise RuntimeError(
                f"{self.name} K-job {n} failed: {result.get('error')}\n"
                f"--- worker.log tail ---\n{self._log_tail()}"
            )
        out_path = output_folder / MAPPING_PROB_FILENAME
        if not out_path.exists():
            raise RuntimeError(f"{self.name} produced no mapping at {out_path}")
        return out_path

    def _log_tail(self, n_chars: int = 2000) -> str:
        try:
            if self._log is not None:
                self._log.flush()
            return self._log_path.read_text()[-n_chars:]
        except Exception:  # noqa: BLE001 — best-effort diagnostics only
            return "(worker log unavailable)"

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        """Signal the worker to exit, then clean up its log and control dir.
        Safe to call more than once."""
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                (self._control / "stop").write_text("")
                try:
                    self._proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
        finally:
            if self._log is not None:
                try:
                    self._log.close()
                except Exception:  # noqa: BLE001
                    pass
                self._log = None
            self._control_tmp.cleanup()
            self._proc = None
