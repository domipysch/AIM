"""
Unified spot->state mapping API (the mapping half of the method).

Given fixed per-state profiles M (from a hard cut of the agglomeration tree) and
the ST data Z, a *mapper* produces a spot->state matrix P (S x K). The merge tree
plays the role of the cluster->state map; only P varies with the mapper and K.

    SpotStateMapper  the contract: .map(Z_shared, M_shared) -> P

Concrete implementations live alongside this module:
    greedy.GreedyMapper   zero-parameter nearest-centroid (one-hot P)
    learned.LearnedMapper gradient-descent soft P

Construct one with ``aim.aim_config.build_mapper``.
"""

from abc import ABC, abstractmethod
import torch


class SpotStateMapper(ABC):
    """A spot->state mapping strategy. Subclasses set ``name`` and implement ``map``."""

    name: str

    @abstractmethod
    def map(self, Z_shared: torch.Tensor, M_shared: torch.Tensor) -> torch.Tensor:
        """Map every ST spot (rows of Z_shared) onto the states whose profiles are
        the rows of M_shared. Both tensors are shared-gene aligned and.
        Returns the spot->state assignment P (S x K): one-hot for greedy, soft
        for learned."""
        raise NotImplementedError

    def config(self) -> dict:
        """Provenance merged into each K's config.yaml. Overridden to add hyperparams."""
        return {"mapping": self.name}
