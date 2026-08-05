"""The ``SpotStateMapper`` API: ``prepare`` stashes the pair's AnnData objects on
the mapper, then ``map`` produces a spot->state matrix P (S x K) for each K cut."""

from abc import ABC, abstractmethod
import torch
import anndata as ad
from aim.adata_schema import UNS_SHARED_GENES
from aim.analysis.utils import to_dense
from ..aggregation import assemble_state_profiles_shared_genes


class SpotStateMapper(ABC):
    """A spot->state mapping strategy; subclasses set ``name`` and implement ``map``."""

    name: str
    adata_sc: ad.AnnData
    adata_st: ad.AnnData

    def prepare(self, adata_sc, adata_st, labels_by_k) -> None:
        """One-time per-pair setup, run once before the K-sweep.

        Stashes the (pre-processed) sc/st AnnData objects on the mapper so ``map``
        and the protected helpers below can reach them without re-passing them per
        K. ``labels_by_k`` maps each swept K to its (L,) subcluster->state cut;
        mappers that need every K's labels up front (e.g. an external reference
        aligner) use it here. Subclasses that override this **must** call
        ``super().prepare(...)`` so the AnnData refs are stored.
        """
        self.adata_sc = adata_sc
        self.adata_st = adata_st

    @abstractmethod
    def map(self, leiden_to_state, k) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map ST spots onto the ``k`` states of the current cut.

        Returns ``(P, confidence)`` where ``P`` is the spot->state matrix (S x K)
        and ``confidence`` is either a (S,) float32 tensor in [0, 1] scoring how
        decisively each spot was assigned, or ``None`` when this mapper does not
        define one (see ``confidence.py``). Higher confidence = more decisive.

        ``leiden_to_state`` is the (L,) subcluster->state cut for this K and ``k``
        the number of states. Implementations build whatever they need (spot
        matrix, state profiles, dispersion) from the AnnData stashed by ``prepare``
        via the protected helpers below.
        """
        raise NotImplementedError

    def _spatial_data_matrix(self) -> torch.Tensor:
        """ST spots on the shared genes as a dense (S x G_shared) float32 tensor.

        Densified because adata_st.X is often sparse and torch.tensor can't
        consume it. K-independent.
        """
        z_shared = to_dense(self.adata_st[:, self.adata_sc.uns[UNS_SHARED_GENES]])
        return torch.tensor(z_shared, dtype=torch.float32)

    def _state_profiles(self, leiden_to_state, k) -> torch.Tensor:
        """Size-weighted per-state raw expression profiles M (k x G_shared)."""
        return assemble_state_profiles_shared_genes(leiden_to_state, k, self.adata_sc)

    def close(self) -> None:
        """Release any resources held for the sweep (e.g. a worker subprocess).
        Default is a no-op; the sweep calls it once after the K-loop."""
