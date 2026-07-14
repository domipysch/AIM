"""Cell-state assignment fractions."""

from __future__ import annotations

import numpy as np


def cell_state_fractions(cell_states: np.ndarray, n_leiden: int) -> dict[int, float]:
    """Return fraction of cells/spots assigned to each state (keys 0 … n_leiden-1).

    n_leiden is the total number of AIM state slots (= L, the number of
    Leiden overclustering clusters — see model.py), not the number of states
    actually in use; bincount needs the full slot range since hard-argmax
    indices can land on any slot.
    """
    counts = np.bincount(cell_states, minlength=n_leiden)
    return {k: float(counts[k]) / len(cell_states) for k in range(n_leiden)}
