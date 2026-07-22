import numpy as np
import logging
import torch
from anndata import AnnData

logger = logging.getLogger(__name__)


def _to_numpy(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def _dense_X(adata: AnnData) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.array(X, dtype=np.float32)


def hard_assignments(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Row-wise argmax → shape (N,)."""
    return _to_numpy(matrix).argmax(axis=1)
