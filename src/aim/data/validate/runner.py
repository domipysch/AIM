"""Orchestration + reporting for ``aim validate``.

Two entry points, one per CLI mode:

* :func:`validate_single_pair` - one scRNA ``.h5ad`` x one ST ``.h5ad``, no index
  CSVs involved (the count/annotation cross-checks are simply skipped).
* :func:`validate_pairs_csv` - every row of a ``pairs.csv``. Each referenced
  dataset is validated once; ``scRNA/index.csv`` and ``ST/index.csv`` are picked
  up next to the h5ad folders when they exist.

Both return a process exit code (0 = no errors) and print a color-coded line per
dataset / pair plus a grouped summary of all findings. :func:`check_pair` is the
same single-pair check without any printing: it returns the findings as data, so
the GUI can show them in its sidebar without re-implementing a single check.

Datasets are loaded lazily and at most one per side is kept in memory: a
``pairs.csv`` with hundreds of ST slices must never hold them all at once.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anndata as ad
import pandas as pd

from .common import colored, status_of
from .validate_pair import validate_pair
from .validate_sc import validate_sc
from .validate_st import validate_st

__all__ = [
    "SubjectFindings",
    "PairFindings",
    "check_pair",
    "validate_single_pair",
    "validate_pairs_csv",
]

SC_DIR_NAME = "scRNA"
ST_DIR_NAME = "ST"
INDEX_NAME = "index.csv"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class _Report:
    """Collects findings per subject and prints them (line now, detail at the end)."""

    def __init__(self) -> None:
        self.errors: dict[str, list[str]] = {}
        self.warns: dict[str, list[str]] = {}

    def add(self, key: str, title: str, errors: list[str], warns: list[str]) -> None:
        if errors:
            self.errors[key] = errors
        if warns:
            self.warns[key] = warns
        status = status_of(errors, warns)
        print(
            f"  {title}: {colored(status, status)} "
            f"(errors={len(errors)}, warns={len(warns)})"
        )

    def finish(self) -> int:
        """Print the grouped summary; return the exit code."""
        if self.warns:
            print("\nWARNINGS:")
            for key, msgs in self.warns.items():
                print(f"  {key}:")
                for msg in msgs:
                    print(f"    - {msg}")

        if self.errors:
            print("\nERRORS:", file=sys.stderr)
            for key, msgs in self.errors.items():
                print(f"  {key}:", file=sys.stderr)
                for msg in msgs:
                    print(f"    - {msg}", file=sys.stderr)
            n = sum(len(m) for m in self.errors.values())
            print(f"\n{n} error(s) in {len(self.errors)} subject(s).", file=sys.stderr)
            return 1

        return 0


# ---------------------------------------------------------------------------
# Single pair: aim validate --scdata ... --stdata ...
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectFindings:
    """What the checks found for one subject (the scRNA, the ST, or the pair)."""

    key: str  # stable id, e.g. "sc:<stem>" / "st:<stem>" / "pair"
    title: str  # display title, e.g. "sc <stem>"
    errors: list[str]
    warns: list[str]

    @property
    def status(self) -> str:
        return status_of(self.errors, self.warns)


@dataclass(frozen=True)
class PairFindings:
    """Every finding for one sc/st pair, in check order (sc, st, pair)."""

    subjects: list[SubjectFindings]

    @property
    def errors(self) -> list[str]:
        return [m for s in self.subjects for m in s.errors]

    @property
    def warns(self) -> list[str]:
        return [m for s in self.subjects for m in s.warns]

    @property
    def status(self) -> str:
        return status_of(self.errors, self.warns)

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def check_pair(
    sc_path: Path,
    st_path: Path,
    on_subject: "Callable[[SubjectFindings], None] | None" = None,
) -> PairFindings:
    """Run every check on one sc/st pair and return the findings; prints nothing.

    The programmatic entry point behind ``aim validate --scdata/--stdata`` - the
    GUI calls this so its sidebar check and the CLI apply the very same checks.
    ``on_subject`` is called as each subject finishes, so a caller can report the
    scRNA verdict while the (possibly large) ST file is still being read.
    """
    sc_path, st_path = Path(sc_path), Path(st_path)
    subjects: list[SubjectFindings] = []

    def _add(key: str, title: str, errors: list[str], warns: list[str]) -> None:
        subject = SubjectFindings(key, title, errors, warns)
        subjects.append(subject)
        if on_subject is not None:
            on_subject(subject)

    sc_errs, sc_warns, adata_sc = validate_sc(sc_path)
    _add(f"sc:{sc_path.stem}", f"sc {sc_path.stem}", sc_errs, sc_warns)
    st_errs, st_warns, adata_st = validate_st(st_path)
    _add(f"st:{st_path.stem}", f"st {st_path.stem}", st_errs, st_warns)

    if adata_sc is None or adata_st is None:
        missing = "scRNA" if adata_sc is None else "ST"
        pair_errs = [f"pair: {missing} dataset could not be loaded - see above"]
        pair_warns: list[str] = []
    else:
        pair_errs, pair_warns = validate_pair(adata_sc, adata_st)
    _add("pair", "pair", pair_errs, pair_warns)

    return PairFindings(subjects)


def validate_single_pair(sc_path: Path, st_path: Path) -> int:
    """Validate one sc/st pair given as two explicit ``.h5ad`` paths."""
    sc_path, st_path = Path(sc_path), Path(st_path)

    print(f"=== {sc_path.stem} x {st_path.stem} ===")
    report = _Report()
    check_pair(
        sc_path,
        st_path,
        on_subject=lambda s: report.add(s.key, s.title, s.errors, s.warns),
    )
    return report.finish()


# ---------------------------------------------------------------------------
# Batch: aim validate --pairs_csv ...
# ---------------------------------------------------------------------------


def validate_pairs_csv(
    pairs_csv: Path,
    sc_dir: Path | None = None,
    st_dir: Path | None = None,
) -> int:
    """Validate every row of ``pairs.csv``.

    ``sc_dir`` / ``st_dir`` default to ``scRNA/`` and ``ST/`` next to the CSV.
    An ``index.csv`` in either folder enables the count / annotation
    cross-checks; without it those checks are skipped (with a warning).
    """
    pairs_csv = Path(pairs_csv)
    if not pairs_csv.exists():
        print(f"ERROR: pairs csv not found: {pairs_csv}", file=sys.stderr)
        return 2

    sc_dir = Path(sc_dir) if sc_dir is not None else pairs_csv.parent / SC_DIR_NAME
    st_dir = Path(st_dir) if st_dir is not None else pairs_csv.parent / ST_DIR_NAME

    try:
        pairs_df = pd.read_csv(pairs_csv, dtype=str).fillna("")
    except Exception as e:
        print(f"ERROR: cannot read {pairs_csv}: {e}", file=sys.stderr)
        return 2

    missing_cols = {"PairID", "scName", "stName"} - set(pairs_df.columns)
    if missing_cols:
        print(
            f"ERROR: {pairs_csv} is missing column(s): {sorted(missing_cols)}",
            file=sys.stderr,
        )
        return 2

    for d in (sc_dir, st_dir):
        if not d.is_dir():
            print(f"ERROR: dataset folder not found: {d}", file=sys.stderr)
            return 2

    sc_index = _load_index(sc_dir)
    st_index = _load_index(st_dir)

    report = _Report()
    print(f"=== {len(pairs_df)} pair(s) from {pairs_csv} ===")

    # One dataset per side stays loaded; pairs.csv is grouped by scName, so each
    # reference is read once per group rather than once per pair.
    sc_cache = _OneSlotCache(sc_dir, sc_index, validate_sc, "sc", report)
    st_cache = _OneSlotCache(st_dir, st_index, validate_st, "st", report)

    for _, row in pairs_df.iterrows():
        pair_id = str(row["PairID"]).strip()
        sc_name = str(row["scName"]).strip()
        st_name = str(row["stName"]).strip()
        title = f"pair {pair_id} ({sc_name} x {st_name})"

        adata_sc = sc_cache.get(sc_name)
        adata_st = st_cache.get(st_name)

        if adata_sc is None or adata_st is None:
            which = f"scRNA '{sc_name}'" if adata_sc is None else f"ST '{st_name}'"
            report.add(
                f"pair:{pair_id}",
                title,
                [f"pair: {which} could not be loaded - see errors above"],
                [],
            )
            continue

        errs, warns = validate_pair(adata_sc, adata_st, row, label=f"pair {pair_id}")
        report.add(f"pair:{pair_id}", title, errs, warns)

    return report.finish()


def _load_index(directory: Path) -> "pd.DataFrame | None":
    """Read ``<directory>/index.csv`` keyed by Name, or None if unusable."""
    path = directory / INDEX_NAME
    if not path.exists():
        print(
            f"NOTE: {path} not found - skipping the index cross-checks for "
            f"{directory.name}/",
            file=sys.stderr,
        )
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("").set_index("Name")
    except Exception as e:
        print(
            f"NOTE: cannot use {path} ({e}) - skipping its cross-checks",
            file=sys.stderr,
        )
        return None


class _OneSlotCache:
    """Validates each dataset once, keeping only the most recent one in memory."""

    def __init__(self, directory, index, validator, side: str, report: _Report) -> None:
        self._dir = directory
        self._index = index
        self._validate = validator
        self._side = side
        self._report = report
        self._name: str | None = None
        self._adata: "ad.AnnData | None" = None
        self._failed: set[str] = set()
        self._reported: set[str] = set()

    def get(self, name: str) -> "ad.AnnData | None":
        if name == self._name:
            return self._adata
        if name in self._failed:
            return None

        # Drop the previous dataset before reading the next one.
        self._name, self._adata = None, None

        index_row = None
        missing_from_index = False
        if self._index is not None:
            if name in self._index.index:
                index_row = self._index.loc[name]
            else:
                missing_from_index = True

        errs, warns, adata = self._validate(
            self._dir / f"{name}.h5ad", index_row, self._side
        )
        if missing_from_index:
            warns.append(
                f"{self._side}: '{name}' has no row in {self._dir.name}/{INDEX_NAME}"
            )

        # A dataset re-read after eviction is validated again but reported once.
        if name not in self._reported:
            self._reported.add(name)
            self._report.add(
                f"{self._side}:{name}", f"{self._side} {name}", errs, warns
            )

        if adata is None:
            self._failed.add(name)
            return None
        self._name, self._adata = name, adata
        return adata
