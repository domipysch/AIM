import torch
from torch import Tensor
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class AIMModel(nn.Module):
    """
    Two-level alignment model.

    Learns two row-stochastic matrices:
        W (S x L)  -> softmax over subclusters -> spot -> Leiden-subcluster deconvolution
        G (L x L)  -> softmax over states      -> Leiden-subcluster -> computed-state merge

    The spot -> state assignment is their product:
        P = W @ G   (S x L)

    Why the product (instead of a free spot->state matrix H):
    Each row of P is a convex combination of G's rows (since W's rows are
    probability vectors). So a spot can only place mass on states that the
    subclusters it is composed of actually map to. Once G sharpens toward
    one-hot rows (via merge_entropy), its rows become vertices on the used
    states, and P is automatically confined to those states — empty/undefined
    states are unreachable by spots, with no explicit masking, no ratchet, and
    full gradient flow between W and G.

    This is what couples the merge (G) to the spatial data: G is only nudged to
    merge subclusters that spots use interchangeably; merging subclusters that
    spots need to keep apart hurts the reconstruction Z' = P @ M(G).

    L = number of Leiden over-clusters = number of state slots.

    The per-cell cell->state matrix B is recovered downstream as B = S_leiden @ G
    (S_leiden = frozen Leiden one-hot, C x L) but is never materialized here.
    """

    def __init__(
        self,
        num_spots_st: int,
        num_leiden_clusters: int,
    ):
        """
        Args:
            num_spots_st:        Number of ST spots (S).
            num_leiden_clusters: Number of Leiden over-clusters (L); also the
                                 number of subclusters and of state slots.
        """
        super(AIMModel, self).__init__()
        self.num_leiden_clusters = num_leiden_clusters

        # W: spot -> Leiden-subcluster logits (S x L); softmax over subclusters
        self.W = nn.Parameter(torch.randn(num_spots_st, num_leiden_clusters))

        # G: Leiden-subcluster -> computed-state logits (L x L); softmax over states
        self.G = nn.Parameter(torch.randn(num_leiden_clusters, num_leiden_clusters))

    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            G: Leiden-subcluster -> computed-state matrix (L x L), rows sum to 1.
            W: spot -> Leiden-subcluster matrix (S x L), rows sum to 1.
            P: spot -> computed-state matrix = W @ G (S x L), rows sum to 1.
        """
        logger.debug("Forward pass")
        G = torch.softmax(self.G, dim=1)
        W = torch.softmax(self.W, dim=1)
        P = torch.matmul(W, G)
        return G, W, P
