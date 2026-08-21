"""Dataset validation - the checks behind ``aim validate``.

* :mod:`~aim.data.validate.validate_sc` - one scRNA reference h5ad
* :mod:`~aim.data.validate.validate_st` - one ST slice h5ad
* :mod:`~aim.data.validate.validate_pair` - the scRNA x ST shared-gene pair
* :mod:`~aim.data.validate.runner` - the two CLI modes (single pair / pairs.csv)
* :mod:`~aim.data.validate.common` - checks shared by the validators

The per-dataset validators take an optional ``index.csv`` row; without one they
run every intrinsic check and skip the bookkeeping cross-checks.
"""

from __future__ import annotations

from .runner import validate_pairs_csv, validate_single_pair
from .validate_pair import validate_pair
from .validate_sc import validate_sc
from .validate_st import validate_st

__all__ = [
    "validate_sc",
    "validate_st",
    "validate_pair",
    "validate_single_pair",
    "validate_pairs_csv",
]
