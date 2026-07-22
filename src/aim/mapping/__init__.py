"""
Spot->state mapping: one unified API, several implementations.

    from aim.mapping import SpotStateMapper, GreedyMapper, LearnedMapper

Add a new strategy by subclassing ``SpotStateMapper``; register it in the
``_MAPPERS`` table in ``aim.aim_config`` so ``build_mapper`` can construct it.
"""

from .base import SpotStateMapper
from .greedy import GreedyMapper
from .learned import LearnedMapper

__all__ = [
    "SpotStateMapper",
    "GreedyMapper",
    "LearnedMapper",
]
