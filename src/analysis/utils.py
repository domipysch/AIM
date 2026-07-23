"""Shared array/AnnData helpers for the post-mapping analysis."""

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
    """Return ``x`` as a dense ndarray, unwrapping AnnData ``.X`` and densifying
    scipy-sparse inputs."""
    if hasattr(x, "X"):
        x = x.X
    if hasattr(x, "toarray"):
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
    """Fraction of cells/spots assigned to each state, keyed 0 .. n_states-1.

    ``n_states`` is the full slot range (= L, the number of subclusters) so
    bincount covers every possible argmax index.
    """
    counts = np.bincount(cell_states, minlength=n_states)
    return {k: float(counts[k]) / len(cell_states) for k in range(n_states)}
