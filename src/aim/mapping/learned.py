"""Learned spot->state mapping: given fixed per-state profiles M, learn a soft P (S x K)
by gradient descent, minimizing spot- and gene-wise cosine distance with a spot_gini
sharpener."""

import logging

import torch

from .base import SpotStateMapper

logger = logging.getLogger(__name__)


def _cosine_distance_loss(
    Z: torch.Tensor, Z_prime: torch.Tensor, dim: int
) -> torch.Tensor:
    """Mean scale-invariant cosine distance (1 - cos) along the given dimension."""
    eps = 1e-8
    dot = torch.sum(Z * Z_prime, dim=dim)
    cos = dot / (torch.norm(Z, p=2, dim=dim) * torch.norm(Z_prime, p=2, dim=dim) + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.mean(torch.clamp(1.0 - cos, min=0.0))


def _gini_weight_at(
    epoch: int, epochs: int, lambda_spot_gini: float, warmup_frac: float
) -> float:
    """Effective spot_gini weight at ``epoch`` under the warmup schedule.

    warmup_frac <= 0 gives a constant ``lambda_spot_gini``. Otherwise the weight is 0
    for the first ``warmup_frac*epochs``, then ramps linearly up to ``lambda_spot_gini``
    at the final epoch, so the sharpener engages only after reconstruction has settled.
    """
    if warmup_frac <= 0.0:
        return lambda_spot_gini
    start = int(warmup_frac * epochs)
    if epoch < start:
        return 0.0
    ramp = (epoch - start + 1) / max(1, epochs - start)
    return lambda_spot_gini * min(1.0, ramp)


class LearnedMapper(SpotStateMapper):
    """Gradient-descent soft mapper: learns P per K by minimizing cosine distance."""

    name = "learned"

    def __init__(
        self,
        epochs: int = 400,
        lr: float = 0.02,
        lambda_spot_gini: float = 1.0,
        spot_gini_warmup_frac: float = 0.5,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.lambda_spot_gini = lambda_spot_gini
        self.spot_gini_warmup_frac = spot_gini_warmup_frac

    def map(self, leiden_to_state, k) -> torch.Tensor:
        """
        Learn the soft spot->state matrix P (S x K) reconstructing the ST data from
        fixed profiles M_shared (K x G_shared).

        Minimizes spot- plus gene-wise cosine distance, plus a per-spot Gini sharpener
        (mean 1 - sum_k P^2, normalized by 1 - 1/K) weighted per the warmup schedule.
        Returns P (S x K) with rows summing to 1.
        """
        Z_shared = self._spatial_data_matrix()
        M_shared = self._state_profiles(leiden_to_state, k)

        n_spots = Z_shared.shape[0]
        n_states = M_shared.shape[0]
        logits = torch.nn.Parameter(torch.randn(n_spots, n_states))

        optimizer = torch.optim.Adam([logits], lr=self.lr)
        gini_norm = (1.0 - 1.0 / n_states) if n_states > 1 else 1.0

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            P = torch.softmax(logits, dim=1)
            Z_prime = torch.matmul(P, M_shared)  # (S x G_shared)
            l_rec_spot = _cosine_distance_loss(Z_shared, Z_prime, dim=1)
            l_rec_gene = _cosine_distance_loss(Z_shared, Z_prime, dim=0)
            gini = (1.0 - torch.sum(P * P, dim=1)).mean() / gini_norm
            w_gini = _gini_weight_at(
                epoch, self.epochs, self.lambda_spot_gini, self.spot_gini_warmup_frac
            )
            loss = l_rec_spot + l_rec_gene
            if w_gini > 0.0 and n_states > 1:
                loss = loss + w_gini * gini
            loss.backward()
            optimizer.step()

        return torch.softmax(logits.detach(), dim=1)

    def config(self) -> dict:
        return {
            "mapping": self.name,
            "epochs": self.epochs,
            "lr": self.lr,
            "lambda_spot_gini": self.lambda_spot_gini,
            "spot_gini_warmup_frac": self.spot_gini_warmup_frac,
        }
