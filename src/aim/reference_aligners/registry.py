"""Registry of the external reference aligners and the runner that executes them.

Adding a reference aligner is two steps:

  1. Write ``aim/reference_aligners/run_<name>.py`` satisfying the CLI contract
     (see ``ReferenceAligner`` / ``run_aligner`` below, or the README section
     "Adding a reference aligner").
  2. Add one ``ReferenceAligner(...)`` line to ``REFERENCE_ALIGNERS``.

The aligner is then available in ``aim run --mapping <name>`` (in either
start-cluster mode), in the GUI sidebar, and as an AIM reference mapper
(``ReferenceMapper``).

Import-light (stdlib only) so it is safe to import from any conda env, including
``aim_env`` where torch/scanpy/aligner deps are not installed.

The wrappers are standalone scripts: they import nothing from ``aim``, and
:func:`run_aligner` executes them **by path** rather than as ``python -m``. An
aligner env therefore needs only its own aligner -- ``spatial-aim`` does not have
to be installed there.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The file every wrapper must write, and every caller reads back.
MAPPING_PROB_FILENAME = "mapping_prob.h5ad"


@dataclass(frozen=True)
class ReferenceAligner:
    """One external aligner: the conda env it runs in and its wrapper script.

    The wrapper is executed **by path** (``python <script_path>``), not as
    ``python -m``, so the aligner's env needs nothing but the aligner itself --
    ``spatial-aim`` does not have to be installed there. A wrapper is therefore a
    standalone script: it must import nothing from the ``aim`` package, only its
    own aligner's libraries.

    The wrapper script MUST implement this CLI contract:

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
    script: str

    @property
    def script_path(self) -> Path:
        """Absolute path of the wrapper script, which sits next to this module."""
        return Path(__file__).resolve().parent / self.script


REFERENCE_ALIGNERS: dict[str, ReferenceAligner] = {
    a.name: a
    for a in (
        ReferenceAligner("tangram", "tangram_env", "run_tangram.py"),
        ReferenceAligner("tacco", "tacco_env", "run_tacco.py"),
        ReferenceAligner("dot", "dot_env", "run_dot.py"),
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

    Executes ``python <wrapper script> --scdata … --stdata … --output_folder …
    --cell_type_key …`` against the aligner's registered env and returns the path
    to the ``mapping_prob.h5ad`` it must produce. Raises ``RuntimeError`` (with the
    stderr tail) if the aligner exits non-zero or leaves no mapping behind.

    One call = one mapping. ``ReferenceMapper`` calls this once per K of a sweep,
    so each K pays a fresh ``conda run`` (env activation + the aligner's imports)
    and re-reads the prepared shared-gene pair; that cold start is the price of
    keeping the wrappers standalone scripts with no ``aim`` dependency.
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
        str(aligner.script_path),
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
        "run_aligner[%s] -> conda run -n %s python %s",
        name,
        aligner.conda_env,
        aligner.script_path,
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
