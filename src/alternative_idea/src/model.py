import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import GCNConv
import logging

logger = logging.getLogger(__name__)


class AlternativeIdeaModel(nn.Module):
    """
    Same as AlternativeIdeaModel, but leave out encoder and decoder on the spatial data.
    Used when L3,4 = 0, decreasing the number of parameters drastically.
    """

    def __init__(
        self,
        num_spots_st: int,
        num_cells_sc: int,
        k: int = 20,
    ):
        """
        Args:
            num_spots_st (S): Number of spots in Z
            num_cells_sc (C): Number of cells in X
            d: Dimension of the embedding space
            k: Upper bound on number of cell states
        """
        super(AlternativeIdeaModel, self).__init__()
        self.num_cells_sc = num_cells_sc

        # U: Spot to cell mapping (S x C)
        # This will become A after Softmax
        self.U = nn.Parameter(torch.randn(num_spots_st, num_cells_sc))

        # V: Cell to cell state mapping (C x K)
        # This will become B after Softmax
        self.V = nn.Parameter(torch.randn(num_cells_sc, k))

    def forward(
        self, z: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            z: Spatial gene features
            edge_index: Graph connectivity

        Returns:
            A: Spot-Cell alignment matrix (S x C)
            B: Cell-State assignment matrix (C x K)
            h: Empty in this case
            M_rec: Empty in this case
            F: Empty in this case
        """
        logger.debug("Forward pass")

        # B. Calculate Alignment Matrices
        # We use Softmax to ensure rows sum to 1.0 (probabilities/proportions)
        # A is the Spot -> Cell mapping (S X C)
        A = torch.softmax(self.U, dim=1)

        # B is the Cell -> State (C x K)
        B = torch.softmax(self.V, dim=1)

        return A, B, Tensor(0), Tensor(0), Tensor(0)
