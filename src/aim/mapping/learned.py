"""
Learned spot->state mapping for the agglomerative method.

Given fixed per-state gene-expression profiles M (assembled from a hard cut of
the agglomeration tree), this learns ONLY the spot->state matrix P (S x K) by
gradient descent — a soft deconvolution of the ST data against the fixed
profiles. The merge tree plays the role of the cluster->state map; only P is
learned here.

The only losses are:
    rec_spot  — spot-wise cosine distance (1 - cos over genes)
    rec_gene  — gene-wise cosine distance (1 - cos over spots)
    spot_gini — quadratic (Gini / Tsallis-2) per-spot sharpening on P, with an
                optional linear warmup so it engages only after reconstruction
                has settled.

``LearnedMapper`` is the ``--mapping learned`` implementation of the
``SpotStateMapper`` API.
"""

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
    """Effective spot_gini weight at a given epoch under the warmup schedule.

    warmup_frac <= 0  -> constant lambda_spot_gini for all epochs (no warmup).
    warmup_frac  > 0  -> weight is 0 for the first warmup_frac*epochs (pure
                         reconstruction), then ramps LINEARLY from 0 up to
                         lambda_spot_gini over the remaining epochs, reaching the
                         full value at the final epoch.

    The warmup exists because sharpening P from a random init makes each spot
    commit to an arbitrary state before reconstruction has found the right one,
    and softmax saturation then freezes that (bad) choice. Letting reconstruction
    settle first, then ramping the sharpener in, sharpens toward the *correct*
    per-spot winner instead.
    """
    if warmup_frac <= 0.0:
        return lambda_spot_gini
    start = int(warmup_frac * epochs)
    if epoch < start:
        return 0.0
    ramp = (epoch - start + 1) / max(1, epochs - start)
    return lambda_spot_gini * min(1.0, ramp)


class LearnedMapper(SpotStateMapper):
    """Gradient-descent soft mapper: trains P per K via ``train_spot_to_state``."""

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

    def map(self, Z_shared: torch.Tensor, M_shared: torch.Tensor) -> torch.Tensor:
        """
        Learn ONLY the spot->state matrix P (S x K) that reconstructs the ST data
        from fixed state profiles M_shared (K x G_shared): minimize spot + gene
        cosine distance, with a quadratic (Gini / Tsallis-2) per-spot sharpener.

        The sharpener is mean per-spot Gini impurity (1 - sum_k P^2), normalised by
        (1 - 1/K). Its gradient is -2*P_k, which keeps pushing leftover second-place
        mass to zero — the strong one-hot lever. spot_gini_warmup_frac schedules it:
        with a value > 0 it stays off for the first warmup_frac*epochs (reconstruction
        settles into the fit-optimal soft solution) and then ramps in linearly, so it
        sharpens toward the correct per-spot winner rather than an arbitrary early
        one. 0.0 = constant weight from the first epoch (no warmup). See _gini_weight_at.

        Returns the spot->state soft assignment P (S x K), rows sum to 1.
        """
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
