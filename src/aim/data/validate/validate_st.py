"""Validate one ST slice ``.h5ad`` against the AIM h5ad contract.

Contract (see ``CLAUDE.md``): ``X`` raw counts, ``var_names`` unique and
uppercase gene symbols, and ``obsm["spatial"]`` of shape ``(n_spots, 2)``.
When an ``ST/index.csv`` row is supplied, its ``SpotCount`` / ``GeneCount``
cells are cross-checked too.
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
    load_X,
)

__all__ = ["validate_st"]


def validate_st(
    h5ad_path: Path,
    index_row: "pd.Series | None" = None,
    label: str = "st",
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

    errors.extend(_check_spatial(adata, label))
    errors.extend(check_gene_names_uppercase(adata, label))
    errors.extend(check_gene_names_unique(adata, label))

    errs, wrns = check_count_matrix(load_X(adata), label)
    errors.extend(errs)
    warns.extend(wrns)

    warns.extend(
        check_index_counts(
            index_row,
            label,
            [
                ("SpotCount", "spots", adata.n_obs),
                ("GeneCount", "genes", adata.n_vars),
            ],
        )
    )

    return errors, warns, adata


def _check_spatial(adata: "ad.AnnData", label: str) -> list[str]:
    """The spatial coordinates every downstream spatial metric relies on."""
    if "spatial" not in adata.obsm:
        return [f"{label}: obsm['spatial'] is missing"]
    spatial = adata.obsm["spatial"]
    if spatial.shape != (adata.n_obs, 2):
        return [f"{label}: obsm['spatial'] shape {spatial.shape} != ({adata.n_obs}, 2)"]
    return []
