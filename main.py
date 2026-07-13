import argparse
import json
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
    arr_to_h5ad,
)
from model import AIMModel
from loss import AIMLoss
from dataset import prepare_tensors_from_input
from evaluate_k.clustering import run_leiden_clustering
from evaluate_k.analysis import run_analysis
from evaluate_k.report import generate_per_k_report

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


def assemble_state_gep(
    G: torch.Tensor,
    expr_sums: torch.Tensor,
    sizes: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Assemble computed-state gene expression profiles from the merge matrix.

        M[k] = (sum_l G[l,k] * expr_sums[l]) / (sum_l G[l,k] * sizes[l])

    i.e. each state centroid is the cell-count-weighted mean of the Leiden
    centroids merged into it.

    Args:
        G:          Leiden-cluster -> computed-state matrix (L x L).
        expr_sums:  Summed expression per Leiden cluster (L x G).
        sizes:      Number of cells per Leiden cluster (L,).

    Returns:
        M: State gene expression profiles (L x G).
    """
    weighted_sum = torch.matmul(G.t(), expr_sums)  # (L x G)
    state_sizes = torch.matmul(G.t(), sizes)  # (L,)
    return weighted_sum / (state_sizes.unsqueeze(1) + eps)


def row_one_hot(mat: torch.Tensor) -> torch.Tensor:
    """Return a row-wise one-hot version of mat (argmax per row -> 1, else 0)."""
    idx = torch.argmax(mat, dim=1, keepdim=True)
    one_hot = torch.zeros_like(mat)
    one_hot.scatter_(1, idx, 1.0)
    return one_hot


def col_one_hot(mat: torch.Tensor) -> torch.Tensor:
    """Return a col-wise one-hot version of mat (argmax per col -> 1, else 0)."""
    idx = torch.argmax(mat, dim=0, keepdim=True)
    one_hot = torch.zeros_like(mat)
    one_hot.scatter_(0, idx, 1.0)
    return one_hot


def aim_compute_mapping(
    adata_sc: AnnData,
    adata_st: AnnData,
    output_folder: Path,
    lr: float,
    epochs: int,
    normalize_and_log: bool,
    leiden_resolution: float,
    lambda_rec_spot: float,
    lambda_rec_gene: float,
    lambda_state_entropy: float,
    lambda_spot_entropy: float,
    lambda_merge_entropy: float,
    lambda_merge_coherence: float,
    verbose_logging: bool,
    device: torch.device,
    save_intermediate: bool = False,
    gpu_limit_gb: int = 6,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
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

    When save_intermediate=True, the following are written to output_folder/intermediate/:
        G.csv / W.csv / P.csv         — merge / spot-subcluster / spot-state matrices (< 0.01 zeroed)
        G_merge.h5ad                  — learned Leiden-subcluster -> state merge matrix (L x L)
        leiden_to_state.csv           — per-cluster dominant state, confidence, size
        B_thresh.h5ad                 — thresholded cell->state soft assignment (entries < 0.1 -> 0)
        M_normalized.h5ad             — state gene expression profiles (L x G_sc)
        cell_states_gep.h5ad          — GEPs for used states only
        p.h5ad / q.h5ad               — mean cell / spot state usage vectors
        cell_state_usage.json         — indices of states with non-zero usage

    Here B (cell -> state, C x L) is the materialized factorization
    B = S_leiden @ G. B_thresh is kept because the grid-search analysis flow reads
    it. The spot->state matrix C = P is NOT saved here — it is mapping_prob.h5ad
    transposed (S x L vs L x S).

    Returns:
        G:              Leiden-subcluster to computed-state matrix (L x L), rows sum to 1.
        W:              Spot to Leiden-subcluster matrix (S x L), rows sum to 1.
        P:              Spot to computed-state matrix = W @ G (S x L), rows sum to 1.
        leiden_labels:  Leiden cluster id per cell (C,), integer.
        expr_sums_full: Summed full-gene expression per Leiden cluster (L x G_sc).
        leiden_sizes:   Number of cells per Leiden cluster (L,).
        losses:         Dict of per-epoch loss values for each component.
    """

    logger.info(f"Using device: {device}")

    # Leiden over-clustering — always computed; K is derived from the cluster count.
    logger.info("Computing Leiden over-clustering...")
    labels_np, _ = run_leiden_clustering(adata_sc, resolution=leiden_resolution)
    leiden_n_clusters = int(labels_np.max()) + 1
    leiden_labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    logger.info(f"Leiden clusters: {leiden_n_clusters}")

    # (Optional) Preprocess data: Normalize & Log-transform
    if normalize_and_log:
        logger.info("Normalize & Log-transform gene expression and spatial data")
        sc.pp.normalize_total(adata_sc)
        sc.pp.normalize_total(adata_st)
        sc.pp.log1p(adata_sc)
        sc.pp.log1p(adata_st)
    else:
        logger.info("Skipping normalize_and_log")

    # Convert input anndata to tensors
    logger.debug("Prepare input tensors for model...")
    X, Z, X_shared, Z_shared = prepare_tensors_from_input(adata_sc, adata_st, device)
    logger.info("Prepared input tensors for model.")
    logger.info(f"Shared genes between scRNA and ST: {X_shared.shape[1]}")

    num_spots, g_st = Z.shape
    num_cells, g_sc = X.shape
    num_genes_shared = X_shared.shape[1]
    logger.debug(f"ST dimensions: num_spots={num_spots}, g_st={g_st}")
    #     logger.debug(f"scRNA dimensions: num_cells={num_cells}, g_sc={g_sc}")
    #     logger.debug(f"Number of genes shared: {num_genes_shared}")
    #
    #     # Fixed Leiden aggregates: expression sums per cluster (shared + full genes)
    # and cluster sizes. The shared-gene sums drive reconstruction; the full-gene
    # sums are used later to impute the complete predicted GEP.
    expr_sums_shared, leiden_sizes = leiden_aggregates(
        X_shared, leiden_labels, leiden_n_clusters
    )
    expr_sums_full, _ = leiden_aggregates(X, leiden_labels, leiden_n_clusters)

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
        g_sc=g_sc,
        g_st=g_st,
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
        lambda_merge_entropy=lambda_merge_entropy,
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
        "merge_entropy": {"weight": lambda_merge_entropy, "values": list()},
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
        losses["merge_entropy"]["values"].append(
            to_scalar(loss_dict.get("merge_entropy"))
        )
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

    if save_intermediate:
        logger.info("Saving intermediate results...")
        folder_intermediate = output_folder / "intermediate"
        folder_intermediate.mkdir(parents=True, exist_ok=True)

        state_names = [f"state_{i}" for i in range(L)]
        leiden_names = [f"leiden_{i}" for i in range(L)]

        # Raw learned matrices as CSV for quick inspection: values < 0.01 zeroed,
        # then rounded to 4 dp.
        G_csv = G.detach().cpu().numpy()
        G_csv[G_csv < 0.01] = 0.0
        pd.DataFrame(G_csv.round(4), index=leiden_names, columns=state_names).to_csv(
            folder_intermediate / "G.csv"  # subcluster -> state
        )
        W_csv = W.detach().cpu().numpy()
        W_csv[W_csv < 0.01] = 0.0
        pd.DataFrame(
            W_csv.round(4), index=adata_st.obs_names.tolist(), columns=leiden_names
        ).to_csv(
            folder_intermediate / "W.csv"
        )  # spot -> subcluster
        P_csv = P.detach().cpu().numpy()
        P_csv[P_csv < 0.01] = 0.0
        pd.DataFrame(
            P_csv.round(4), index=adata_st.obs_names.tolist(), columns=state_names
        ).to_csv(
            folder_intermediate / "P_matrix.csv"
        )  # spot -> state

        # Materialized cell->state B = S_leiden @ G (C x L). The spot->state
        # matrix is P = W @ G (S x L).
        B = G[leiden_labels]  # (C x L)

        B_thresh = B.detach().cpu().clone()
        B_thresh[B_thresh < 0.1] = 0.0
        arr_to_h5ad(
            B_thresh.numpy(),
            folder_intermediate / "B_thresh.h5ad",
            obs_names=adata_sc.obs_names.tolist(),
            var_names=state_names,
        )

        # Spot->state P is not written as h5ad here: it equals mapping_prob.h5ad
        # transposed (S x L vs L x S). Kept in-memory only for the usage report.
        C_thresh = P.detach().cpu().clone()
        C_thresh[C_thresh < 0.1] = 0.0

        B_used_mask = B_thresh.sum(dim=0) > 0
        C_used_mask = C_thresh.sum(dim=0) > 0
        state_usage = {
            "B_used_count": int(B_used_mask.sum().item()),
            "B_used_indices": B_used_mask.nonzero(as_tuple=False).squeeze(1).tolist(),
            "C_used_count": int(C_used_mask.sum().item()),
            "C_used_indices": C_used_mask.nonzero(as_tuple=False).squeeze(1).tolist(),
        }
        with open(folder_intermediate / "cell_state_usage.json", "w") as f:
            json.dump(state_usage, f, indent=2)

        # Spots should only use states that cells populate; warn (do not crash a
        # completed run) if a spot maps to an empty state.
        if not set(state_usage["C_used_indices"]).issubset(
            set(state_usage["B_used_indices"])
        ):
            logger.warning(
                "C_used_indices is not a subset of B_used_indices: "
                f"C={state_usage['C_used_indices']}, B={state_usage['B_used_indices']}"
            )

        p = torch.mean(B, dim=0)
        p_np = p.detach().cpu().numpy()
        pd.DataFrame(p_np.round(2)).to_csv(folder_intermediate / "p.csv", index=False)
        arr_to_h5ad(
            p_np.reshape(1, -1),
            folder_intermediate / "p.h5ad",
            obs_names=["mean_cell_state_usage"],
            var_names=state_names,
        )

        q = torch.mean(P, dim=0)
        q_np = q.detach().cpu().numpy()
        pd.DataFrame(q_np.round(2)).to_csv(folder_intermediate / "q.csv", index=False)
        arr_to_h5ad(
            q_np.reshape(1, -1),
            folder_intermediate / "q.h5ad",
            obs_names=["mean_spot_state_usage"],
            var_names=state_names,
        )

        # State gene expression profiles (full genes)
        M = assemble_state_gep(G, expr_sums_full, leiden_sizes)
        M_np = M.detach().cpu().numpy()
        arr_to_h5ad(
            M_np,
            folder_intermediate / "M_normalized.h5ad",
            obs_names=state_names,
            var_names=adata_sc.var_names.tolist(),
        )

        c_used_idx = state_usage["C_used_indices"]
        cell_states = M_np[c_used_idx, :]
        arr_to_h5ad(
            cell_states,
            folder_intermediate / "cell_states_gep.h5ad",
            obs_names=[f"state_{i}" for i in c_used_idx],
            var_names=adata_sc.var_names.tolist(),
        )

        # G: Leiden-cluster -> computed-state merge matrix (L x L). This is the
        # actual learned reference-side object; B above is just its cell-expanded
        # view (B = G[leiden_labels]).
        leiden_names = [f"leiden_{i}" for i in range(L)]
        G_np = G.detach().cpu().numpy()
        arr_to_h5ad(
            G_np,
            folder_intermediate / "G_merge.h5ad",
            obs_names=leiden_names,
            var_names=state_names,
        )

        # Compact merge map: each Leiden cluster's dominant state, the weight on
        # it (confidence — 1.0 means a perfectly one-hot / sharp merge row), and
        # the cluster size. Handy for spotting diffuse (unsharpened) rows.
        merge_state = G_np.argmax(axis=1)
        merge_confidence = G_np.max(axis=1)
        pd.DataFrame(
            {
                "leiden_cluster": list(range(L)),
                "n_cells": leiden_sizes.detach().cpu().numpy().astype(int),
                "state": merge_state,
                "confidence": merge_confidence.round(3),
            }
        ).to_csv(folder_intermediate / "leiden_to_state.csv", index=False)

    logger.info("Alignment complete.")
    return G, W, P, leiden_labels, expr_sums_full, leiden_sizes, losses


def compute_gene_expression_prediction(
    spot_to_state: torch.Tensor,
    merge: torch.Tensor,
    expr_sums_full: torch.Tensor,
    leiden_sizes: torch.Tensor,
    adata_sc: AnnData,
    adata_st: AnnData,
    deterministic_mapping: bool,
    torch_device: torch.device,
) -> AnnData:
    """
    Predict spot-level gene expression from the learned mapping.

    Computes Z' = C @ M, where C is the spot->state matrix P and M is the
    computed-state gene expression profile assembled from the merge matrix G
    and the fixed full-gene Leiden aggregates.

    In deterministic mode, both G (Leiden->state) and P (spot->state) are
    replaced by their row-wise argmax (one-hot) before computing the prediction,
    so each Leiden cluster maps to one state and each spot maps to one state.

    Args:
        spot_to_state:   P — soft spot-to-state matrix (S x L), rows sum to 1.
        merge:           G — soft Leiden-to-state matrix (L x L), rows sum to 1.
        expr_sums_full:  Summed full-gene expression per Leiden cluster (L x G_sc).
        leiden_sizes:    Number of cells per Leiden cluster (L,).
        adata_sc:        scRNA-seq reference (C x G_sc), used for gene ids only.
        adata_st:        Spatial transcriptomics data (S x G_st), used for spot ids only.
        deterministic_mapping: If True, hard-assign G and P before predicting.
        torch_device:    Device for intermediate tensor computation.

    Returns:
        AnnData of shape (G_sc x S): predicted expression matrix,
        obs_names = gene symbols, var_names = spot IDs.
    """
    if deterministic_mapping:
        G_use = row_one_hot(merge)
        C = row_one_hot(spot_to_state)
    else:
        G_use = merge
        C = spot_to_state

    M = assemble_state_gep(G_use, expr_sums_full, leiden_sizes)  # (L x G_sc)
    predicted_spot_expressions = torch.matmul(C, M)  # (S x G_sc)

    # Transpose to G x S
    predicted_spot_expressions = predicted_spot_expressions.T  # now G x S
    assert predicted_spot_expressions.shape == (
        adata_sc.n_vars,
        adata_st.n_obs,
    ), "dims passen nicht"

    # Create AnnData object for predicted spot expressions
    adata_result = AnnData(X=predicted_spot_expressions.detach().cpu().numpy())
    adata_result.obs_names = adata_sc.var_names
    adata_result.var_names = adata_st.obs_names

    return adata_result


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
    lambda_spot_entropy: float = 0.08,
    lambda_merge_entropy: float = 1.0,
    lambda_merge_coherence: float = 0.5,
    store_intermediate: bool = False,
    skip_analysis: bool = False,
    verbose_logging: bool = False,
    gpu_limit_gb: int = 6,
) -> tuple[AnnData, AnnData, AnnData, dict]:
    """
    Load data, run mapping, compute both probabilistic and deterministic GEPs,
    and write all outputs to output_folder.

    Outputs written:
        gep_prob.h5ad      — probabilistic predicted GEP (G x S)
        gep_det.h5ad       — deterministic predicted GEP (G x S)
        mapping_prob.h5ad  — soft spot-to-state map (L x S)
        mapping_det.h5ad   — hard spot-to-state map (L x S, one-hot columns)
        loss/              — per-epoch loss curves and final values CSV
        intermediate/      — B, C, M matrices (only when store_intermediate=True)

    Returns:
        gep_prob:                Probabilistic GEP AnnData (G x S).
        gep_det:                 Deterministic GEP AnnData (G x S).
        cell_to_cell_type_adata: Cell-to-state assignment matrix as AnnData (C x L).
        losses_after_last_epoch: Dict of final loss component values.
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
    G, W, P, leiden_labels, expr_sums_full, leiden_sizes, losses = aim_compute_mapping(
        adata_sc=adata_sc.copy(),
        adata_st=adata_st.copy(),
        output_folder=output_folder,
        lr=lr,
        epochs=epochs,
        normalize_and_log=normalize_and_log,
        leiden_resolution=leiden_resolution,
        lambda_rec_spot=lambda_rec_spot,
        lambda_rec_gene=lambda_rec_gene,
        lambda_state_entropy=lambda_state_entropy,
        lambda_spot_entropy=lambda_spot_entropy,
        lambda_merge_entropy=lambda_merge_entropy,
        lambda_merge_coherence=lambda_merge_coherence,
        verbose_logging=verbose_logging,
        device=device,
        save_intermediate=store_intermediate,
        gpu_limit_gb=gpu_limit_gb,
    )
    logger.info("Obtained spot-to-state mapping.")
    L = G.shape[1]
    assert P.shape == (adata_st.n_obs, L), "dims passen nicht"

    # Materialize cell->state assignment B = S_leiden @ G (C x L) for downstream analysis
    B = G[leiden_labels]  # (C x L)
    cell_to_cell_type_adata = AnnData(X=B.detach().cpu().numpy())
    cell_to_cell_type_adata.obs_names = adata_sc.obs_names
    cell_to_cell_type_adata.var_names = [f"Type {n}" for n in range(L)]

    # Save loss logs and plots
    losses_after_last_epoch = dump_loss_logs(losses, output_folder)
    create_loss_plots(losses, output_folder / "loss")

    # Compute and save probabilistic GEP
    adata_prediction_prob = compute_gene_expression_prediction(
        P,
        G,
        expr_sums_full,
        leiden_sizes,
        adata_sc,
        adata_st,
        False,
        device,
    )
    gep_prob_path = output_folder / "gep_prob.h5ad"
    adata_prediction_prob.write_h5ad(gep_prob_path)
    logger.info(f"Saved prob GEP to {gep_prob_path}")

    state_names = [f"state_{i}" for i in range(L)]
    mapping_np = P.detach().cpu().numpy().T  # L x S (state x spot)
    mapping_prob_path = output_folder / "mapping_prob.h5ad"
    AnnData(
        X=mapping_np.astype(np.float32),
        obs=pd.DataFrame(index=state_names),
        var=pd.DataFrame(index=adata_st.obs_names),
    ).write_h5ad(mapping_prob_path)
    logger.info(f"Saved prob mapping to {mapping_prob_path}")

    # Compute and save deterministic GEP
    logger.info("Apply deterministic mapping & compute prediction")
    adata_prediction_det = compute_gene_expression_prediction(
        P,
        G,
        expr_sums_full,
        leiden_sizes,
        adata_sc,
        adata_st,
        True,
        device,
    )
    gep_det_path = output_folder / "gep_det.h5ad"
    adata_prediction_det.write_h5ad(gep_det_path)
    logger.info(f"Saved det GEP to {gep_det_path}")

    one_hot = row_one_hot(P).detach().cpu().numpy().T.astype(np.uint8)  # L x S
    mapping_det_path = output_folder / "mapping_det.h5ad"
    AnnData(
        X=one_hot,
        obs=pd.DataFrame(index=state_names),
        var=pd.DataFrame(index=adata_st.obs_names),
    ).write_h5ad(mapping_det_path)
    logger.info(f"Saved det mapping to {mapping_det_path}")

    # Post-mapping analysis and report
    if not skip_analysis:
        try:
            B_np = B.detach().cpu().numpy()  # C x L
            C_np = P.detach().cpu().numpy()  # S x L
            B_np[B_np < 0.1] = 0.0
            C_np[C_np < 0.1] = 0.0
            K_analysis = L
            analysis_dir = output_folder / "analysis"
            run_analysis(
                adata_sc=adata_sc,
                adata_st=adata_st,
                B=B_np,
                C=C_np,
                output_dir=analysis_dir,
                K=K_analysis,
                leiden_resolution=leiden_resolution,
            )
            generate_per_k_report(analysis_dir, K_analysis, run_id="0")
            logger.info("Analysis report written to %s", analysis_dir)
        except Exception as _analysis_exc:
            logger.error("Analysis failed (outputs already saved): %s", _analysis_exc)

    return (
        adata_prediction_prob,
        adata_prediction_det,
        cell_to_cell_type_adata,
        losses_after_last_epoch,
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
        help="Normalize and log-transform input data before training",
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
    parser.add_argument("--lambda_state_entropy", type=float, default=0.1)
    parser.add_argument("--lambda_spot_entropy", type=float, default=0.08)
    parser.add_argument("--lambda_merge_entropy", type=float, default=1.0)
    parser.add_argument("--lambda_merge_coherence", type=float, default=0.5)
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
                "training": {
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "normalize_and_log": args.normalize_and_log,
                    "reference_leiden_clustering_resolution": args.leiden_resolution,
                },
                "loss_weights": {
                    "lambda_rec_spot": args.lambda_rec_spot,
                    "lambda_rec_gene": args.lambda_rec_gene,
                    "lambda_state_entropy": args.lambda_state_entropy,
                    "lambda_spot_entropy": args.lambda_spot_entropy,
                    "lambda_merge_entropy": args.lambda_merge_entropy,
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
        lambda_merge_entropy=args.lambda_merge_entropy,
        lambda_merge_coherence=args.lambda_merge_coherence,
        verbose_logging=(args.logging == "verbose"),
        store_intermediate=True,
        gpu_limit_gb=args.gpu_limit_gb,
    )
