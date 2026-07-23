"""AIM (Annotation Independent Mapping): over-cluster an scRNA reference, agglomerate
the subclusters into K states, and map ST spots onto those states across a K sweep.
Public surface: ``AIMConfig``, ``MAPPING_CHOICES``, ``run``, ``SpotStateMapper``."""

from .aim_config import MAPPING_CHOICES, AIMConfig
from .mapping import SpotStateMapper
from .sweep import run

__all__ = [
    "AIMConfig",
    "MAPPING_CHOICES",
    "run",
    "SpotStateMapper",
]
