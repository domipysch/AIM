"""AIM (Annotation Independent Mapping): over-cluster an scRNA reference, agglomerate
the subclusters into K states, and map ST spots onto those states across a K sweep.
Public surface: ``AIMConfig``, ``MAPPING_CHOICES``, ``run``, ``SpotStateMapper``.

These names are exposed lazily (PEP 562) so that importing ``aim``, or a
submodule such as ``aim.reference_aligners.run_tangram`` from inside a reference
aligner's own conda env, does not pull in the heavy sweep/analysis stack
(scanpy, squidpy, ...), which is only present in ``aim_env``. The heavy modules
load on first attribute access instead.
"""

from importlib import import_module
from typing import TYPE_CHECKING

_LAZY = {
    "AIMConfig": ("aim.aim_config", "AIMConfig"),
    "MAPPING_CHOICES": ("aim.aim_config", "MAPPING_CHOICES"),
    "SpotStateMapper": ("aim.mapping", "SpotStateMapper"),
    "run": ("aim.sweep", "run"),
}

__all__ = ["AIMConfig", "MAPPING_CHOICES", "run", "SpotStateMapper"]


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # let type checkers and IDEs see the real symbols
    from aim.aim_config import MAPPING_CHOICES, AIMConfig
    from aim.mapping import SpotStateMapper
    from aim.sweep import run
