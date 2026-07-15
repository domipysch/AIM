import argparse
import sys
from pathlib import Path
from typing import Any
import anndata
import pandas as pd
import yaml
import scanpy as sc
import torch
import torch.optim as optim
import logging
import numpy as np
from anndata import AnnData
from utils import (
    create_loss_plots,
    dump_loss_logs,
    estimate_gpu_memory_gb,
)
from model import AIMModel
from loss import AIMLoss
from dataset import prepare_tensors_from_input
from analysis.clustering import run_leiden_clustering

logger = logging.getLogger(__name__)


def leiden_aggregates(
    expr: torch.Tensor, labels: torch.Tensor, n_clusters: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate per-Leiden-cluster expression sums and cluster sizes.

    Args:
        expr:       Expression matrix (C x G), on device.
        labels:     Leiden cluster id per cell (C,), integer, on device.
        n_clusters: Number of Leiden clusters (L).

    Returns:
        expr_sums:  Summed expression per cluster (L x G).
        sizes:      Number of cells per cluster (L,).
    """
    expr_sums = torch.zeros(
        n_clusters, expr.shape[1], device=expr.device, dtype=expr.dtype
    )
    expr_sums.index_add_(0, labels, expr)
    sizes = torch.bincount(labels, minlength=n_clusters).to(expr.dtype)
    return expr_sums, sizes


def aim_compute_mapping(
    adata_sc: AnnData,
    adata_st: AnnData,
    lr: float,
    epochs: int,
    normalize_and_log: bool,
    leiden_resolution: float,
    lambda_rec_spot: float,
    lambda_rec_gene: float,
    lambda_state_entropy: float,
    lambda_spot_entropy: float,
    lambda_spot_gini: float,
    lambda_merge_entropy: float,
    lambda_merge_gini: float,
    lambda_merge_coherence: float,
    verbose_logging: bool,
    device: torch.device,
    gpu_limit_gb: int = 6,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict,
]:
    """
    Run the full optimization loop and return the learned mappings.

    K (number of computed states) always equals the number of Leiden clusters at
    leiden_resolution. Leiden over-clustering is computed once before training;
    its per-cluster expression sums and sizes are fixed and used to assemble the
    computed-state gene expression profiles.

    The model learns two row-stochastic matrices:
        W (S x L) — spot -> Leiden-subcluster deconvolution
        G (L x L) — Leiden-subcluster -> computed-state merge
    and their product P = W @ G (S x L) is the spot -> computed-state assignment.

    Returns:
        G:              Leiden-subcluster to computed-state matrix (L x L), rows sum to 1.
        W:              Spot to Leiden-subcluster matrix (S x L), rows sum to 1.
        P:              Spot to computed-state matrix = W @ G (S x L), rows sum to 1.
        leiden_labels:  Leiden cluster id per cell (C,), integer.
        losses:         Dict of per-epoch loss values for each component.
    """

    logger.info(f"Using device: {device}")

    # Leiden over-clustering — always computed; K is derived from the cluster count.
    logger.info("Computing Leiden over-clustering...")
    labels_np, _ = run_leiden_clustering(adata_sc, resolution=leiden_resolution)
    leiden_n_clusters = int(labels_np.max()) + 1
    leiden_labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    logger.info(f"Leiden clusters: {leiden_n_clusters}")

    # (Optional) Preprocess data: total-count normalization, optionally + log1p.
    if normalize_and_log:
        sc.pp.normalize_total(adata_sc)
        sc.pp.normalize_total(adata_st)
        sc.pp.log1p(adata_sc)
        sc.pp.log1p(adata_st)
    else:
        logger.info("Skipping normalize_and_log")

    # Convert input anndata to tensors (shared genes only — the full, non-shared
    # matrices are never needed by the model, so they are not materialized).
    logger.debug("Prepare input tensors for model...")
    X_shared, Z_shared = prepare_tensors_from_input(adata_sc, adata_st, device)
    logger.info("Prepared input tensors for model.")
    logger.info(f"Shared genes between scRNA and ST: {X_shared.shape[1]}")

    num_spots, g_st = adata_st.n_obs, adata_st.n_vars
    num_cells, g_sc = adata_sc.n_obs, adata_sc.n_vars
    num_genes_shared = X_shared.shape[1]
    logger.debug(f"ST dimensions: num_spots={num_spots}, g_st={g_st}")

    # Fixed Leiden aggregates: shared-gene expression sums per cluster and cluster
    # sizes. These drive the reconstruction loss and the merge-coherence distance.
    expr_sums_shared, leiden_sizes = leiden_aggregates(
        X_shared, leiden_labels, leiden_n_clusters
    )

    # Pairwise shared-gene cosine distance between Leiden centroids (fixed).
    # Drives the merge-coherence loss: subclusters close here are hard to tell
    # apart from the shared-gene spatial view, so merging them is cheap.
    centroids_shared = expr_sums_shared / (leiden_sizes.unsqueeze(1) + 1e-8)
    centroids_norm = centroids_shared / (
        centroids_shared.norm(dim=1, keepdim=True) + 1e-8
    )
    leiden_dist = (1.0 - torch.matmul(centroids_norm, centroids_norm.t())).clamp(
        min=0.0
    )
    leiden_dist.fill_diagonal_(0.0)

    # GPU memory guard
    L = leiden_n_clusters
    estimated_gb = estimate_gpu_memory_gb(
        num_cells=num_cells,
        num_spots=num_spots,
        num_genes_shared=num_genes_shared,
        n_states=L,
    )
    logger.info(f"Estimated GPU memory requirement: {estimated_gb:.2f} GB")
    if estimated_gb > gpu_limit_gb:
        logger.error(
            f"Estimated GPU memory ({estimated_gb:.2f} GB) exceeds the {gpu_limit_gb} GB limit. "
            f"Aborting to prevent OOM. Pass --gpu_limit_gb with a higher value to override."
        )
        sys.exit(1)

    model = AIMModel(
        num_spots_st=num_spots,
        num_leiden_clusters=leiden_n_clusters,
    ).to(device)

    # Initialize Loss and Optimizer
    loss = AIMLoss(
        leiden_expr_sums=expr_sums_shared,
        leiden_sizes=leiden_sizes,
        leiden_dist=leiden_dist,
        lambda_rec_spot=lambda_rec_spot,
        lambda_rec_gene=lambda_rec_gene,
        lambda_state_entropy=lambda_state_entropy,
        lambda_spot_entropy=lambda_spot_entropy,
        lambda_spot_gini=lambda_spot_gini,
        lambda_merge_entropy=lambda_merge_entropy,
        lambda_merge_gini=lambda_merge_gini,
        lambda_merge_coherence=lambda_merge_coherence,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    logger.info("Starting optimization loop")
    model.train()

    # Collect per-epoch losses for plotting: individual components + total
    losses: dict[str, Any] = {
        "total-weighted": list(),
        "rec_spot": {"weight": lambda_rec_spot, "values": list()},
        "rec_gene": {"weight": lambda_rec_gene, "values": list()},
        "state_entropy": {"weight": lambda_state_entropy, "values": list()},
        "spot_entropy": {"weight": lambda_spot_entropy, "values": list()},
        "spot_gini": {"weight": lambda_spot_gini, "values": list()},
        "merge_entropy": {"weight": lambda_merge_entropy, "values": list()},
        "merge_gini": {"weight": lambda_merge_gini, "values": list()},
        "merge_coherence": {"weight": lambda_merge_coherence, "values": list()},
    }

    def to_scalar(t):
        try:
            if torch.is_tensor(t):
                return float(t.detach().cpu().item())
            else:
                return float(t)
        except Exception:
            return float(t)

    assert epochs > 0
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        G, W, P = model()

        # Calculate segmented losses
        loss_dict = loss(G=G, P=P, Z_shared=Z_shared)
        total_loss = loss_dict["loss"]

        # Optional gradient diagnostic (verbose mode only, every 100 epochs)
        if epoch % 100 == 0 and verbose_logging:
            logger.debug(f"\n--- Gradient Analysis (Epoch {epoch}) ---")
            for name, loss_val in loss_dict.items():
                if name == "loss":
                    continue

                # Compute per-term gradient norm for diagnostics
                model.zero_grad()
                loss_val.backward(retain_graph=True)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1000
                )
                logger.debug(
                    f"Term: {name:15} | Loss: {loss_val.item():.4f} | Grad-Norm: {grad_norm:.4f}"
                )

            # Reset before the actual backward pass below
            model.zero_grad()

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Collect loss values
        losses["total-weighted"].append(to_scalar(total_loss))
        losses["rec_spot"]["values"].append(to_scalar(loss_dict.get("rec_spot")))
        losses["rec_gene"]["values"].append(to_scalar(loss_dict.get("rec_gene")))
        losses["state_entropy"]["values"].append(
            to_scalar(loss_dict.get("state_entropy"))
        )
        losses["spot_entropy"]["values"].append(
            to_scalar(loss_dict.get("spot_entropy"))
        )
        losses["spot_gini"]["values"].append(to_scalar(loss_dict.get("spot_gini")))
        losses["merge_entropy"]["values"].append(
            to_scalar(loss_dict.get("merge_entropy"))
        )
        losses["merge_gini"]["values"].append(to_scalar(loss_dict.get("merge_gini")))
        losses["merge_coherence"]["values"].append(
            to_scalar(loss_dict.get("merge_coherence"))
        )
        # Logging: verbose -> log every epoch at DEBUG, normal -> log every 10 epochs at INFO
        if verbose_logging:
            logger.debug(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")
        else:
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")

    # Final detached matrices for saving and downstream use.
    with torch.no_grad():
        G, W, P = model()

    logger.info("Alignment complete.")
    return G, W, P, leiden_labels, losses


def main(
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
    lr: float = 0.008,
    epochs: int = 1000,
    normalize_and_log: bool = False,
    leiden_resolution: float = 3.0,
    lambda_rec_spot: float = 0.5,
    lambda_rec_gene: float = 0.5,
    lambda_state_entropy: float = 0.1,
    lambda_spot_entropy: float = 0.0,
    lambda_spot_gini: float = 0.5,
    lambda_merge_entropy: float = 0.0,
    lambda_merge_gini: float = 1.0,
    lambda_merge_coherence: float = 0.5,
    verbose_logging: bool = False,
    gpu_limit_gb: int = 6,
):
    """
    Load data, run mapping, and write all outputs to output_folder.

    Post-mapping analysis is decoupled from this runner — it is not computed
    here. Once these outputs are written, run it separately via
    analysis.run_from_output (reads the files below from disk).

    Outputs written:
        mapping_prob.h5ad      — soft spot-to-state map (obs=spots, var=computed states)
        leiden_merge_prob.h5ad — soft Leiden-subcluster-to-state merge matrix G
                                 (obs=Leiden subclusters, var=computed states)
        leiden_overclustering.h5ad     — per-cell Leiden overclustering label (obs=cells,
                                 obs["leiden_cluster"], no var); combine with
                                 leiden_merge_prob.h5ad (G) to recover the computed-state
                                 assignment per cell
        loss/                  — per-epoch loss curves and final values CSV
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Step 1: Load data
    logger.info("Load input scRNA and ST data...")
    adata_sc = anndata.read_h5ad(sc_path)  # C x G
    adata_st = anndata.read_h5ad(st_path)  # S x G
    logger.info("Loaded input scRNA and ST data.")

    # Step 2: Map data using AIM
    G, W, P, leiden_labels, losses = aim_compute_mapping(
        adata_sc=adata_sc.copy(),
        adata_st=adata_st.copy(),
        lr=lr,
        epochs=epochs,
        normalize_and_log=normalize_and_log,
        leiden_resolution=leiden_resolution,
        lambda_rec_spot=lambda_rec_spot,
        lambda_rec_gene=lambda_rec_gene,
        lambda_state_entropy=lambda_state_entropy,
        lambda_spot_entropy=lambda_spot_entropy,
        lambda_spot_gini=lambda_spot_gini,
        lambda_merge_entropy=lambda_merge_entropy,
        lambda_merge_gini=lambda_merge_gini,
        lambda_merge_coherence=lambda_merge_coherence,
        verbose_logging=verbose_logging,
        device=device,
        gpu_limit_gb=gpu_limit_gb,
    )
    logger.info("Obtained spot-to-state mapping.")
    L = G.shape[1]
    assert P.shape == (adata_st.n_obs, L), "dims passen nicht"
    assert G.shape == (L, L), "dims passen nicht"
    assert W.shape == (adata_st.n_obs, L), "dims passen nicht"

    # Save loss logs and plots
    losses_after_last_epoch = dump_loss_logs(losses, output_folder)
    create_loss_plots(losses, output_folder / "loss")

    state_names = [f"state_{i}" for i in range(L)]
    leiden_names = [f"leiden_{i}" for i in range(L)]

    # Save mapping: obs=spots, var=computed states (S x L)
    mapping_prob_path = output_folder / "mapping_prob.h5ad"
    AnnData(
        X=P.detach().cpu().numpy().astype(np.float32),
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=state_names),
    ).write_h5ad(mapping_prob_path)
    logger.info(f"Saved prob mapping to {mapping_prob_path}")

    # Save Leiden-subcluster merge matrix: obs=Leiden subclusters, var=computed
    # states (L x L) — this is G, the learned Leiden-subcluster -> computed-state
    # merge matrix.
    leiden_merge_prob_path = output_folder / "leiden_merge_prob.h5ad"
    AnnData(
        X=G.detach().cpu().numpy().astype(np.float32),
        obs=pd.DataFrame(index=leiden_names),
        var=pd.DataFrame(index=state_names),
    ).write_h5ad(leiden_merge_prob_path)
    logger.info(f"Saved Leiden merge matrix to {leiden_merge_prob_path}")

    # Save clusters: obs=cells, no var — obs["leiden_cluster"] names each cell's Leiden
    # overclustering label (the hard Leiden assignment, not the merged computed
    # state). Combine with leiden_merge_prob.h5ad (G) downstream to recover the
    # computed-state assignment per cell.
    leiden_labels_np = leiden_labels.detach().cpu().numpy()
    cell_cluster_names = [leiden_names[i] for i in leiden_labels_np]
    leiden_overclustering_path = output_folder / "leiden_overclustering.h5ad"
    AnnData(
        X=np.zeros((len(cell_cluster_names), 0), dtype=np.float32),
        obs=pd.DataFrame(
            {"leiden_cluster": cell_cluster_names}, index=adata_sc.obs_names
        ),
    ).write_h5ad(leiden_overclustering_path)
    logger.info(
        f"Saved cell-to-leiden-cluster assignment to {leiden_overclustering_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run AIM alignment on a dataset folder"
    )
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Folder where all outputs are written (config.yaml is saved here)",
    )
    parser.add_argument(
        "--logging",
        choices=["normal", "verbose"],
        default="normal",
        help="Logging verbosity. Use 'verbose' for more logs.",
    )
    parser.add_argument(
        "--gpu_limit_gb",
        type=int,
        default=48,
        help="GPU memory limit in GB. Abort if estimated usage exceeds this value.",
    )
    # training
    parser.add_argument("--lr", type=float, default=0.008, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--normalize_and_log",
        action="store_true",
        default=False,
        help="Total-count normalize and log1p-transform input data before training",
    )
    parser.add_argument(
        "--leiden_resolution",
        type=float,
        default=3.0,
        help="Leiden clustering resolution; also sets the number of computed states (training.reference_leiden_clustering_resolution)",
    )
    # loss weights
    parser.add_argument("--lambda_rec_spot", type=float, default=0.5)
    parser.add_argument("--lambda_rec_gene", type=float, default=0.5)
    parser.add_argument("--lambda_state_entropy", type=float, default=1.0)
    parser.add_argument("--lambda_spot_entropy", type=float, default=0.0)
    parser.add_argument("--lambda_spot_gini", type=float, default=0.5)
    parser.add_argument("--lambda_merge_entropy", type=float, default=0.2)
    parser.add_argument("--lambda_merge_gini", type=float, default=0.0)
    parser.add_argument("--lambda_merge_coherence", type=float, default=0.0)
    args = parser.parse_args()

    level = logging.DEBUG if args.logging == "verbose" else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(level)

    # Write params to config.yaml for provenance, then pass directly into main()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    with open(args.output_folder / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "lr": args.lr,
                "epochs": args.epochs,
                "normalize_and_log": args.normalize_and_log,
                "reference_leiden_clustering_resolution": args.leiden_resolution,
                "loss_weights": {
                    "lambda_rec_spot": args.lambda_rec_spot,
                    "lambda_rec_gene": args.lambda_rec_gene,
                    "lambda_state_entropy": args.lambda_state_entropy,
                    "lambda_spot_entropy": args.lambda_spot_entropy,
                    "lambda_spot_gini": args.lambda_spot_gini,
                    "lambda_merge_entropy": args.lambda_merge_entropy,
                    "lambda_merge_gini": args.lambda_merge_gini,
                    "lambda_merge_coherence": args.lambda_merge_coherence,
                },
            },
            f,
            sort_keys=False,
        )

    main(
        args.scdata,
        args.stdata,
        output_folder=args.output_folder,
        lr=args.lr,
        epochs=args.epochs,
        normalize_and_log=args.normalize_and_log,
        leiden_resolution=args.leiden_resolution,
        lambda_rec_spot=args.lambda_rec_spot,
        lambda_rec_gene=args.lambda_rec_gene,
        lambda_state_entropy=args.lambda_state_entropy,
        lambda_spot_entropy=args.lambda_spot_entropy,
        lambda_spot_gini=args.lambda_spot_gini,
        lambda_merge_entropy=args.lambda_merge_entropy,
        lambda_merge_gini=args.lambda_merge_gini,
        lambda_merge_coherence=args.lambda_merge_coherence,
        verbose_logging=(args.logging == "verbose"),
        gpu_limit_gb=args.gpu_limit_gb,
    )
