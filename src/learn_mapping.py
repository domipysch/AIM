"""
Learned spot->state mapping for the agglomerative method.

Given fixed per-state gene-expression profiles M (assembled from a hard cut of
the agglomeration tree in main.py), this learns ONLY the spot->state matrix P
(S x K) by gradient descent — a soft deconvolution of the ST data against the
fixed profiles. The merge tree plays the role of the cluster->state map; only P
is learned here.

The only losses are:
    rec_spot  — spot-wise cosine distance (1 - cos over genes)
    rec_gene  — gene-wise cosine distance (1 - cos over spots)
    spot_gini — quadratic (Gini / Tsallis-2) per-spot sharpening on P, with an
                optional linear warmup so it engages only after reconstruction
                has settled.

This module is imported by main.py when it is run with --mapping learned. The
zero-parameter alternative (--mapping greedy, nearest-centroid) needs none of
this and lives in main.py directly.
"""

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

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


def gini_weight_at(
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


def train_spot_to_state(
    M_shared: torch.Tensor,
    Z_shared: torch.Tensor,
    epochs: int,
    lr: float,
    lambda_spot_gini: float,
    device: torch.device,
    spot_gini_warmup_frac: float = 0.5,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
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
    one. 0.0 = constant weight from the first epoch (no warmup). See gini_weight_at.

    Returns:
        P:       spot->state soft assignment (S x K), rows sum to 1.
        history: per-epoch unweighted values of each loss component
                 (rec_spot, rec_gene, spot_gini) plus the effective gini weight
                 actually applied that epoch (spot_gini_weight).
    """
    n_spots = Z_shared.shape[0]
    n_states = M_shared.shape[0]
    logits = torch.nn.Parameter(torch.randn(n_spots, n_states, device=device))

    optimizer = torch.optim.Adam([logits], lr=lr)
    gini_norm = (1.0 - 1.0 / n_states) if n_states > 1 else 1.0

    history: dict[str, list[float]] = {
        "rec_spot": [],
        "rec_gene": [],
        "spot_gini": [],
        "spot_gini_weight": [],
    }
    for epoch in range(epochs):
        optimizer.zero_grad()
        P = torch.softmax(logits, dim=1)
        Z_prime = torch.matmul(P, M_shared)  # (S x G_shared)
        l_rec_spot = _cosine_distance_loss(Z_shared, Z_prime, dim=1)
        l_rec_gene = _cosine_distance_loss(Z_shared, Z_prime, dim=0)
        gini = (1.0 - torch.sum(P * P, dim=1)).mean() / gini_norm
        w_gini = gini_weight_at(epoch, epochs, lambda_spot_gini, spot_gini_warmup_frac)
        loss = l_rec_spot + l_rec_gene
        if w_gini > 0.0 and n_states > 1:
            loss = loss + w_gini * gini
        loss.backward()
        optimizer.step()
        history["rec_spot"].append(float(l_rec_spot.detach()))
        history["rec_gene"].append(float(l_rec_gene.detach()))
        history["spot_gini"].append(float(gini.detach()))
        history["spot_gini_weight"].append(float(w_gini))

    return torch.softmax(logits.detach(), dim=1), history


def plot_loss_curves(history: dict[str, list[float]], out_path: Path, k: int) -> None:
    """Plot the per-epoch deconvolution loss components for one K (learned mode)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(len(history["rec_spot"]))
    ax.plot(epochs, history["rec_spot"], label="rec_spot (1 - cos, per spot)")
    ax.plot(epochs, history["rec_gene"], label="rec_gene (1 - cos, per gene)")
    ax.plot(epochs, history["spot_gini"], label="spot_gini (normalised)")
    if any(w > 0 for w in history.get("spot_gini_weight", [])):
        ax.plot(
            epochs,
            history["spot_gini_weight"],
            ls=":",
            color="0.5",
            label="spot_gini weight (schedule)",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss component (unweighted)")
    ax.set_title(f"Deconvolution loss components (K = {k})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
