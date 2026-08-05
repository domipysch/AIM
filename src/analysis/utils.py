"""Shared array/AnnData helpers for the post-mapping analysis."""

from __future__ import annotations
import numpy as np

__all__ = ["to_dense"]


def to_dense(x, dtype=np.float32) -> np.ndarray:
    """Return ``x`` as a dense ndarray, unwrapping AnnData ``.X`` and densifying
    scipy-sparse inputs."""
    if hasattr(x, "X"):
        x = x.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=dtype)
