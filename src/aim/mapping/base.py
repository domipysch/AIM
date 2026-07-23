"""The ``SpotStateMapper`` API: given ST data Z and fixed per-state profiles M,
produce a spot->state matrix P (S x K)."""

from abc import ABC, abstractmethod
import torch


class SpotStateMapper(ABC):
    """A spot->state mapping strategy; subclasses set ``name`` and implement ``map``."""

    name: str

    def prepare(self, adata_sc, adata_st, labels_by_k) -> None:
        """Optional one-time per-pair setup, run once before the K-sweep.

        Default is a no-op; mappers that need the full AnnData objects or every
        K's state labels up front (e.g. an external reference aligner) override
        this. ``labels_by_k`` maps each swept K to its (L,) subcluster->state cut.
        """
        return None

    @abstractmethod
    def map(self, Z_shared: torch.Tensor, M_shared: torch.Tensor) -> torch.Tensor:
        """Map ST spots (rows of Z_shared) onto the states (rows of M_shared), both
        shared-gene aligned; returns the spot->state matrix P (S x K)."""
        raise NotImplementedError

    def config(self) -> dict:
        """Config dict describing this mapper; subclasses add their hyperparameters."""
        return {"mapping": self.name}
