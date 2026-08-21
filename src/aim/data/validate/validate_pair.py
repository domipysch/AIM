"""Validate an scRNA x ST pair: the shared-gene intersection AIM maps through.

Both AnnData objects must already have passed :func:`validate_sc` /
:func:`validate_st`. When a ``pairs.csv`` row is supplied, its
``NumberSharedGenes`` cell is cross-checked too.
"""

from __future__ import annotations

import anndata as ad
import pandas as pd
from scipy.sparse import issparse

from .common import is_intable, to_int_safe

__all__ = ["validate_pair"]


def validate_pair(
    adata_sc: "ad.AnnData",
    adata_st: "ad.AnnData",
    index_row: "pd.Series | None" = None,
    label: str = "pair",
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warns)`` for the pair as such (not for either dataset)."""
    errors: list[str] = []
    warns: list[str] = []

    shared = adata_sc.var_names.intersection(adata_st.var_names)
    n_shared = len(shared)

    if n_shared <= 10:
        warns.append(f"{label}: only {n_shared} shared genes (expected > 10)")

    if index_row is not None:
        val = index_row.get("NumberSharedGenes", "")
        if is_intable(val) and to_int_safe(val) != n_shared:
            warns.append(
                f"{label}: NumberSharedGenes={val} != actual shared genes={n_shared}"
            )

    if n_shared > 0:
        st_sub = adata_st[:, shared].X
        if issparse(st_sub):
            st_sub = st_sub.toarray()
        n_zero = (st_sub.sum(axis=1) == 0).sum()
        if n_zero > 0:
            warns.append(
                f"{label}: {n_zero} ST spot(s) are all-zero after the gene "
                f"intersection"
            )

    return errors, warns
