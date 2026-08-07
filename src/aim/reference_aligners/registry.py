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


def _conda_from_root(root: Path) -> str | None:
    """The conda launcher inside a conda install rooted at ``root``, or ``None``.

    Layout differs by OS: Windows keeps it in ``Scripts\\conda.exe`` (or
    ``condabin\\conda.bat``); POSIX in ``bin/conda`` (or ``condabin/conda``).
    Returns the first candidate that exists as a file."""
    if os.name == "nt":
        candidates = [
            root / "Scripts" / "conda.exe",
            root / "condabin" / "conda.bat",
            root / "condabin" / "conda.exe",
        ]
    else:
        candidates = [root / "bin" / "conda", root / "condabin" / "conda"]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _prefer_real_exe(exe: str) -> str:
    """Prefer a real conda executable over a Windows batch wrapper.

    ``CONDA_EXE`` (and a ``PATHEXT`` hit from :func:`shutil.which`) frequently
    points at ``condabin\\conda.bat``. Running that ``.bat`` via ``subprocess``
    with no shell makes it recurse into itself until Windows aborts it
    ("BATCH RECURSION exceeds stack limit", exit 255, empty stdout) — so every
    ``conda env list`` comes back empty and all aligners look uninstalled.

    ``Scripts\\conda.exe`` from the same install root is a normal executable
    that ``subprocess`` runs cleanly, so resolve to it when the launcher is a
    ``.bat``. Non-``.bat`` paths (and ``.bat`` paths with no sibling ``.exe``)
    are returned unchanged."""
    p = Path(exe)
    if p.suffix.lower() != ".bat":
        return exe
    # condabin\conda.bat -> install root is its grandparent; the real launcher
    # lives in <root>\Scripts\conda.exe.
    root = p.parent.parent if p.parent.name.lower() == "condabin" else p.parent
    real = root / "Scripts" / "conda.exe"
    return str(real) if real.is_file() else exe


def _search_conda_locations() -> str | None:
    """Best-effort discovery of a conda install when it is not on ``PATH``.

    Checks, in order:

    1. ``CONDA_PREFIX`` / ``CONDA_ROOT`` — set when a conda env is active but
       conda's launcher dirs were never added to ``PATH`` (a common state when
       the process is started from an IDE, a service, or a plain shell). From
       ``CONDA_PREFIX`` we climb to the install root (an env lives at
       ``<root>/envs/<name>``; base is the root itself).
    2. The standard per-user install roots for miniforge/miniconda/anaconda —
       both directly under the home directory and (on Windows) under
       ``%LOCALAPPDATA%``, where the miniforge installer defaults.

    Returns the launcher path, or ``None`` if nothing is found."""
    roots: list[Path] = []

    for var in ("CONDA_PREFIX", "CONDA_ROOT"):
        val = os.environ.get(var)
        if val:
            prefix = Path(val)
            # An active env sits at <root>/envs/<name>; base sits at <root>.
            if prefix.parent.name == "envs":
                roots.append(prefix.parent.parent)
            roots.append(prefix)

    home = Path.home()
    bases = [home]
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        bases.append(Path(localappdata))
    for base in bases:
        for name in ("miniforge3", "miniconda3", "anaconda3", "mambaforge"):
            roots.append(base / name)

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        exe = _conda_from_root(root)
        if exe:
            return exe
    return None


def conda_exe() -> str:
    """Locate a conda executable, OS-independently, in priority order:

    1. the ``CONDA_EXE`` environment variable, if set (an explicit override —
       the full path to a conda/mamba executable), then
    2. ``conda`` found on ``PATH`` via :func:`shutil.which` (which applies
       ``PATHEXT`` on Windows, so it resolves ``conda.exe``/``conda.bat``
       transparently, and needs no extension elsewhere), then
    3. a fallback search of ``CONDA_PREFIX``/``CONDA_ROOT`` and the standard
       miniforge/miniconda/anaconda install roots (see
       :func:`_search_conda_locations`) — because a process started outside an
       activated conda shell (IDE run configs, ``aim gui`` launched from a plain
       terminal) often has neither ``CONDA_EXE`` set nor conda on ``PATH`` even
       though conda is installed.

    Whatever is found is passed through :func:`_prefer_real_exe`, which swaps a
    Windows ``conda.bat`` wrapper for the sibling ``Scripts\\conda.exe`` (the
    ``.bat`` recurses to death under ``subprocess``; see that function).

    Raises ``RuntimeError`` if none yields a conda. That is an expected,
    non-fatal state: spatial-aim can be pip-installed into a plain venv with no
    conda at all — only the external reference aligners (Tangram/TACCO/DOT, which
    run in their own conda envs) are unavailable then; the in-process mappers do
    not call this."""
    exe = (
        os.environ.get("CONDA_EXE")
        or shutil.which("conda")
        or _search_conda_locations()
    )
    if not exe:
        raise RuntimeError(
            "conda not found: the reference aligners run in their own conda env "
            "via `conda run`. Install conda and either set CONDA_EXE to the "
            "conda executable or put conda on PATH. (Not required for the "
            "in-process mappers or for a plain pip/venv install.)"
        )
    return _prefer_real_exe(exe)


def _first_json_object(text: str) -> dict | None:
    """Decode the first JSON object in ``text``, ignoring anything before the
    opening ``{`` or after the closing ``}``.

    conda occasionally leaks notices/warnings into the ``--json`` stdout stream,
    so a plain ``json.loads`` of the whole capture raises ``JSONDecodeError:
    Extra data``. ``raw_decode`` parses just the object and stops. Returns
    ``None`` if no JSON object can be found."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def available_conda_envs() -> set[str]:
    """Names of the conda envs currently installed, from ``conda env list
    --json``. Empty set if conda cannot be located or its output cannot be
    parsed.

    Reads each env's ``name`` from ``envs_details`` (falling back to the path
    basename for older conda that omits it). The JSON is extracted with
    :func:`_first_json_object` rather than ``json.loads`` because conda may print
    extra text around the JSON."""
    try:
        exe = conda_exe()
    except RuntimeError:
        return set()
    try:
        # Capture RAW BYTES (no text/encoding): conda's output is not reliably
        # UTF-8 — on a localized Windows it can contain OEM/ANSI-codepage bytes
        # (e.g. 0x81) that are invalid UTF-8. If subprocess decoded in its reader
        # thread, that byte would crash the thread mid-read, break conda's stdout
        # pipe, and surface as a non-zero exit (255) — making every aligner look
        # uninstalled. Reading bytes and decoding ourselves with errors="replace"
        # can never raise; the env-path JSON is pure ASCII, so any replaced bytes
        # only ever land in surrounding notice text. No check=True: parse whatever
        # came back, so a conda that exits non-zero but still emitted JSON works.
        proc = subprocess.run(
            [exe, "env", "list", "--json"],
            capture_output=True,
        )
    except OSError:
        return set()

    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    data = _first_json_object(stdout)
    if data is None:
        return set()

    details = data.get("envs_details") or {}
    names = {d.get("name") for d in details.values() if d.get("name")}
    if names:
        return names
    # Older conda without envs_details: derive from the path basename.
    return {os.path.basename(p.rstrip("/\\")) for p in data.get("envs", [])}


def available_reference_aligners() -> list[str]:
    """Reference aligner names whose registered conda env is currently
    installed, in ``REFERENCE_ALIGNERS`` order. Empty if none can be found."""
    envs = available_conda_envs()
    return [name for name, a in REFERENCE_ALIGNERS.items() if a.conda_env in envs]


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
        # UTF-8 + errors="replace": don't let a non-cp1252 byte in the aligner's
        # output turn into a UnicodeDecodeError that masks the real failure (see
        # available_conda_envs).
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
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
            # errors="replace": the worker log holds raw child output, which may
            # contain non-locale-codepage bytes; don't let decoding it swallow
            # the diagnostics we're trying to show.
            return self._log_path.read_text(encoding="utf-8", errors="replace")[
                -n_chars:
            ]
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
