"""Shared helpers for the post-mapping analysis.

Small utilities used across the ``analysis`` package (and by
``reference_aligners/mapping_analysis``, which imports ``run_pca_neighbors_umap``
and ``fmt_nonzero_4`` from here). Pure array/AnnData helpers with no method- or
disk-layout-specific logic — the actual metric computations live in ``metrics``.

``to_dense`` (the project-wide densification idiom) lives here; ``metrics`` and
``plots`` import it from this module.
"""

from __future__ import annotations
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "to_dense",
    "to_numpy",
    "hard_assignments",
    "cell_state_fractions",
]


def to_dense(x, dtype=np.float32) -> np.ndarray:
    """Return ``x`` as a dense ``ndarray``.

    Accepts an AnnData (its ``.X`` is used), a scipy-sparse matrix, or anything
    array-like. Sparse inputs are materialized via ``.toarray()``.
    """
    if hasattr(x, "X"):  # AnnData / AnnData view
        x = x.X
    if hasattr(x, "toarray"):  # scipy sparse
        x = x.toarray()
    return np.asarray(x, dtype=dtype)


def to_numpy(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Detach a torch tensor to CPU numpy, or pass an array-like through ``np.asarray``."""
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def hard_assignments(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Row-wise argmax -> shape (N,)."""
    return to_numpy(matrix).argmax(axis=1)


def cell_state_fractions(cell_states: np.ndarray, n_states: int) -> dict[int, float]:
    """Fraction of cells/spots assigned to each state (keys 0 .. n_states-1).

    ``n_states`` is the full slot range (= L, the number of subclusters), not the
    number of states actually in use: bincount needs the full range since
    hard-argmax indices can land on any slot.
    """
    counts = np.bincount(cell_states, minlength=n_states)
    return {k: float(counts[k]) / len(cell_states) for k in range(n_states)}
