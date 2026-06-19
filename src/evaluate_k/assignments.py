"""Cell-state assignment fractions."""

from __future__ import annotations

import numpy as np


def cell_state_fractions(cell_states: np.ndarray, K: int) -> dict[int, float]:
    """Return fraction of cells/spots assigned to each state (keys 0 … K-1)."""
    counts = np.bincount(cell_states, minlength=K)
    return {k: float(counts[k]) / len(cell_states) for k in range(K)}
