"""Validate one scRNA reference ``.h5ad`` against the AIM h5ad contract.

Contract (see ``CLAUDE.md``): ``X`` raw counts, ``obs`` cell metadata with at
least one cell-type column, ``var_names`` unique and uppercase gene symbols.
When an ``scRNA/index.csv`` row is supplied, its ``CellTypeKey*`` /
``NumberCellTypes*`` / ``CellCount`` / ``GeneCount`` cells are cross-checked too.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from .common import (
    check_count_matrix,
    check_gene_names_unique,
    check_gene_names_uppercase,
    check_index_counts,
    is_intable,
    load_X,
    to_int_safe,
)

__all__ = ["validate_sc"]


def validate_sc(
    h5ad_path: Path,
    index_row: "pd.Series | None" = None,
    label: str = "sc",
) -> tuple[list[str], list[str], "ad.AnnData | None"]:
    """Return ``(errors, warns, adata)``; ``adata`` is None if it could not be read."""
    errors: list[str] = []
    warns: list[str] = []

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        return [f"{label}: missing file {h5ad_path}"], warns, None

    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        return [f"{label}: cannot read {h5ad_path} ({e})"], warns, None

    errors.extend(check_gene_names_uppercase(adata, label))
    errors.extend(check_gene_names_unique(adata, label))

    errs, wrns = check_count_matrix(load_X(adata), label)
    errors.extend(errs)
    warns.extend(wrns)

    errs, wrns = _check_cell_type_keys(adata, index_row, label)
    errors.extend(errs)
    warns.extend(wrns)

    warns.extend(
        check_index_counts(
            index_row,
            label,
            [
                ("CellCount", "cells", adata.n_obs),
                ("GeneCount", "genes", adata.n_vars),
            ],
        )
    )

    return errors, warns, adata


def _check_cell_type_keys(
    adata: "ad.AnnData",
    index_row: "pd.Series | None",
    label: str,
) -> tuple[list[str], list[str]]:
    """CellTypeKey0/1/2 must exist in obs; NumberCellTypes* must match its cardinality."""
    if index_row is None:
        return [], []

    errors: list[str] = []
    warns: list[str] = []
    for i in range(3):
        key_col = f"CellTypeKey{i}"
        num_col = f"NumberCellTypes{i}"
        key_val = str(index_row.get(key_col, "")).strip()
        if not key_val:
            break
        if key_val not in adata.obs.columns:
            errors.append(
                f"{label}: '{key_val}' (from {key_col}) not found in obs columns"
            )
            continue
        actual_n = adata.obs[key_val].nunique()
        num_val = str(index_row.get(num_col, "")).strip()
        if not num_val:
            warns.append(f"{label}: {num_col} not set (expected count for '{key_val}')")
        elif is_intable(num_val) and to_int_safe(num_val) != actual_n:
            warns.append(
                f"{label}: {num_col}={num_val} != actual unique values "
                f"in '{key_val}'={actual_n}"
            )
    return errors, warns
