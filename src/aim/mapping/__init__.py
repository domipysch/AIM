"""Spot->state mapping: the ``SpotStateMapper`` API and its ``NearestMapper`` and
``LearnedMapper`` implementations."""

from .base import SpotStateMapper
from .nearest import NearestMapper
from .nearest_scaled import NearestScaledMapper
from .nearest_euclidean import NearestEuclideanMapper
from .nearest_euclidean_scaled import NearestEuclideanScaledMapper
from .learned import LearnedMapper
from .majority_vote import MajorityVoteMapper
from .majority_vote_euclidean import MajorityVoteEuclideanMapper
from .reference import ReferenceMapper

__all__ = [
    "SpotStateMapper",
    "NearestMapper",
    "NearestScaledMapper",
    "NearestEuclideanMapper",
    "NearestEuclideanScaledMapper",
    "LearnedMapper",
    "MajorityVoteMapper",
    "MajorityVoteEuclideanMapper",
    "ReferenceMapper",
]
