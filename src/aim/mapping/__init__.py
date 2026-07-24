"""Spot->state mapping: the ``SpotStateMapper`` API and its ``NearestMapper`` and
``LearnedMapper`` implementations."""

from .base import SpotStateMapper
from .nearest import NearestMapper
from .nearest_scaled import NearestScaledMapper
from .learned import LearnedMapper
from .majority_vote import MajorityVoteMapper
from .reference import ReferenceMapper

__all__ = [
    "SpotStateMapper",
    "NearestMapper",
    "NearestScaledMapper",
    "LearnedMapper",
    "MajorityVoteMapper",
    "ReferenceMapper",
]
