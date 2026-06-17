from typing import Dict
import numpy as np
import pandas as pd
import logging
import json
import torch
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level array helpers (shared across training and evaluate_k)
# ---------------------------------------------------------------------------


def _to_numpy(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def _dense_X(adata: AnnData) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.array(X, dtype=np.float32)


def hard_assignments(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Row-wise argmax → shape (N,)."""
    return _to_numpy(matrix).argmax(axis=1)


# ---------------------------------------------------------------------------
# Shared scanpy PCA + neighbors pipeline
# ---------------------------------------------------------------------------


def run_pca_neighbors_umap(
    adata: AnnData,
    n_comps: int = 30,
    n_neighbors: int = 15,
    skip_umap: bool = False,
) -> None:
    """In-place: optional normalize/log1p → PCA → neighbors → optional UMAP."""

    # Normalize before computing PCAs
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n = min(n_comps, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    if not skip_umap:
        sc.tl.umap(adata)


def fmt_nonzero_4(x: float) -> str:
    """
    Format a numeric value for display to cap at up to four decimal places.

    Args:
        x: Input value (float)
    Returns:
        str: Formatted string
    """
    if pd.isna(x):
        return ""
    try:
        xf = float(x)
    except Exception:
        raise Exception("Input value is not convertible to float")
    if xf == 0.0:
        return "0.0"
    return f"{xf:.4f}"


def dump_loss_logs(losses, config_path) -> dict:

    losses_after_last_epoch = {}
    for comp in (
        "rec_spot",
        "rec_gene",
        "rec_state",
        "clust",
        "clust_intra",
        "clust_inter",
        "state_entropy",
        "spot_entropy",
        "soft_contingency",
    ):
        comp_vals = losses.get(comp, {})
        val = None
        if isinstance(comp_vals, dict):
            vals_list = comp_vals.get("values", [])
            weight = comp_vals.get("weight")
            if len(vals_list) > 0:
                val = vals_list[-1]

        # Round unweighted final value to 2 decimals for clarity (handle None)
        losses_after_last_epoch[f"{comp}"] = (
            round(float(val), 2) if val is not None else None
        )

    loss_dir = config_path.parent / "loss"
    loss_dir.mkdir(parents=True, exist_ok=True)
    df_end = pd.DataFrame([losses_after_last_epoch])
    df_end.to_csv(loss_dir / "losses_end.csv", index=False)
    logger.info(f"Saved final loss values to {loss_dir / 'losses_end.csv'}")

    losses_all = {
        comp: losses[comp]["values"]
        for comp in losses
        if isinstance(losses.get(comp), dict) and "values" in losses[comp]
    }
    with open(loss_dir / "losses_all.json", "w") as f:
        json.dump(losses_all, f, indent=2)
    logger.info(f"Saved all loss values to {loss_dir / 'losses_all.json'}")

    return losses_after_last_epoch


def create_loss_plots(losses, loss_dir):

    loss_dir.mkdir(parents=True, exist_ok=True)

    loss_fig_path = loss_dir / "loss–curves-weighted.pdf"
    plt.figure()
    # Plot individual components + total
    epochs = list(range(len(losses["total-weighted"])))
    plt.plot(
        epochs,
        losses["total-weighted"],
        label="total-weighted",
        linewidth=2,
        color="black",
    )
    plt.plot(
        epochs,
        list(v * losses["rec_spot"]["weight"] for v in losses["rec_spot"]["values"]),
        label="rec_spot-weighted",
    )
    plt.plot(
        epochs,
        list(v * losses["rec_gene"]["weight"] for v in losses["rec_gene"]["values"]),
        label="rec_gene-weighted",
    )
    if "rec_state" in losses:
        plt.plot(
            epochs,
            list(
                v * losses["rec_state"]["weight"] for v in losses["rec_state"]["values"]
            ),
            label="rec_state-weighted",
        )
    if "clust" in losses:
        plt.plot(
            epochs,
            list(v * losses["clust"]["weight"] for v in losses["clust"]["values"]),
            label="clust-weighted",
        )
    if "clust_intra" in losses:
        plt.plot(
            epochs,
            list(
                v * losses["clust_intra"]["weight"]
                for v in losses["clust_intra"]["values"]
            ),
            label="clust_intra-weighted",
        )
    if "clust_inter" in losses:
        plt.plot(
            epochs,
            list(
                v * losses["clust_inter"]["weight"]
                for v in losses["clust_inter"]["values"]
            ),
            label="clust_inter-weighted",
        )
    plt.plot(
        epochs,
        list(
            v * losses["state_entropy"]["weight"]
            for v in losses["state_entropy"]["values"]
        ),
        label="state_entropy-weighted",
    )
    plt.plot(
        epochs,
        list(
            v * losses["spot_entropy"]["weight"]
            for v in losses["spot_entropy"]["values"]
        ),
        label="spot_entropy-weighted",
    )
    if "soft_contingency" in losses:
        plt.plot(
            epochs,
            list(
                v * losses["soft_contingency"]["weight"]
                for v in losses["soft_contingency"]["values"]
            ),
            label="soft_contingency-weighted",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (components + total)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(loss_fig_path))
    plt.close()
    logger.info(f"Saved loss curve to {loss_fig_path}")

    num_epochs = len(losses.get("total-weighted", []))

    # Components (order for plotting)
    components = (
        "rec_spot",
        "rec_gene",
        *(("rec_state",) if "rec_state" in losses else ()),
        *(("clust",) if "clust" in losses else ()),
        "state_entropy",
        "spot_entropy",
        *(("clust_intra",) if "clust_intra" in losses else ()),
        *(("clust_inter",) if "clust_inter" in losses else ()),
        *(("soft_contingency",) if "soft_contingency" in losses else ()),
    )

    # Ensure epochs list is available
    epochs = list(range(num_epochs))

    # Save one plot per loss component
    for comp in components:
        plt.figure()
        y = losses[comp]["values"]
        plt.plot(epochs, y, label=comp, linewidth=1)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title(f"Loss curve: L_{comp} - unweighted")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_path = loss_dir / f"{comp}.pdf"
        plt.savefig(str(out_path))
        plt.close()
        logger.info(f"Saved per-loss plot to {out_path}")
