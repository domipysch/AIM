import torch
from torch import Tensor
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class AIMModel(nn.Module):
    """
    Alignment model with two learnable mapping matrices.

    Learns two parameter matrices projected to row-stochastic matrices via softmax:
        U (S × C) → A: spot-to-cell soft assignment
        V (C × K) → B: cell-to-state soft assignment

    The combined spot-to-state matrix C = A @ B is used by the loss functions.
    """

    def __init__(
        self,
        num_spots_st: int,
        num_cells_sc: int,
        n_states: int,
    ):
        """
        Args:
            num_spots_st: Number of ST spots (S).
            num_cells_sc: Number of scRNA cells (C).
            n_states: Number of cell states (K), derived from Leiden clustering.
        """
        super(AIMModel, self).__init__()
        self.num_cells_sc = num_cells_sc

        # U: Spot to cell mapping (S x C)
        # This will become A after Softmax
        self.U = nn.Parameter(torch.randn(num_spots_st, num_cells_sc))

        # V: Cell to cell state mapping (C x K)
        # This will become B after Softmax
        self.V = nn.Parameter(torch.randn(num_cells_sc, n_states))

    def forward(self) -> tuple[Tensor, Tensor]:
        """
        Returns:
            A: Spot-Cell alignment matrix (S x C)
            B: Cell-State assignment matrix (C x K)
        """
        logger.debug("Forward pass")

        # B. Calculate Alignment Matrices
        # We use Softmax to ensure rows sum to 1.0 (probabilities/proportions)
        # A is the Spot -> Cell mapping (S X C)
        A = torch.softmax(self.U, dim=1)

        # B is the Cell -> State (C x K)
        B = torch.softmax(self.V, dim=1)

        return A, B
