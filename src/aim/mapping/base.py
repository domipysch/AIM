"""The ``SpotStateMapper`` API: given ST data Z and fixed per-state profiles M,
produce a spot->state matrix P (S x K)."""

from abc import ABC, abstractmethod
import torch


class SpotStateMapper(ABC):
    """A spot->state mapping strategy; subclasses set ``name`` and implement ``map``."""

    name: str

    @abstractmethod
    def map(self, Z_shared: torch.Tensor, M_shared: torch.Tensor) -> torch.Tensor:
        """Map ST spots (rows of Z_shared) onto the states (rows of M_shared), both
        shared-gene aligned; returns the spot->state matrix P (S x K)."""
        raise NotImplementedError

    def config(self) -> dict:
        """Config dict describing this mapper; subclasses add their hyperparameters."""
        return {"mapping": self.name}
