"""Spot->state mapping: the ``SpotStateMapper`` API and its ``NearestCentroidMapper``,
``WANNMapper``, and ``ReferenceMapper`` implementations."""

from .base import SpotStateMapper
from .nearest_centroid import NearestCentroidMapper
from .wann import WANNMapper
from .reference import ReferenceMapper

__all__ = [
    "SpotStateMapper",
    "NearestCentroidMapper",
    "WANNMapper",
    "ReferenceMapper",
]
