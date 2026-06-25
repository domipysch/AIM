import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional
import anndata
import pandas as pd
import yaml
import scanpy as sc
import torch
import torch.optim as optim
import logging
import numpy as np
import scipy.sparse as sp
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

# from sc_embedding import compute_sc_embedding

logger = logging.getLogger(__name__)


def aim_compute_mapping(
    adata_sc: AnnData,
    adata_st: AnnData,
    output_folder: Path,
    K: int,
    lr: float,
    epochs: int,
    normalize_and_log: bool,
    leiden_resolution: float,
    # sc_embedding_method: str,
    # sc_embedding_d: int,
    lambda_rec_spot: float,
    lambda_rec_gene: float,
    # lambda_clust_intra: float,
    # lambda_clust_inter: float,
    lambda_state_entropy: float,
    lambda_spot_entropy: float,
    lambda_soft_contingency: float,
    verbose_logging: bool,
    device: torch.device,
    save_intermediate: bool = False,
    gpu_limit_gb: int = 6,
    # sc_data_dir: Optional[Path] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Run the full optimization loop and return the learned mappings.

    Leiden over-clustering is computed once before training (when
    lambda_soft_contingency > 0) and used as a fixed supervision signal.

    When save_intermediate=True, the following are written to output_folder/intermediate/:
        B_thresh.h5ad, C_thresh.h5ad  — thresholded soft assignments (entries < 0.1 → 0)
        B_argmax.h5ad, C_argmax.h5ad  — hard (one-hot) assignments
        M_normalized.h5ad             — state gene expression profiles (K × G_sc)
        cell_states_gep.h5ad          — GEPs for used states only
        p.h5ad / q.h5ad               — mean cell / spot state usage vectors
        cell_state_usage.json         — indices of states with non-zero usage

    Returns:
        A: Spot-to-cell soft assignment matrix (S × C), rows sum to 1.
        B: Cell-to-state soft assignment matrix (C × K), rows sum to 1.
        losses: Dict of per-epoch loss values for each component.
    """

    logger.info(f"Using device: {device}")

    # Leiden over-clustering for soft contingency loss (once, before training)
    # Must happen before prepare_tensors_from_input so adata_sc is still available
    leiden_labels_tensor, leiden_n_clusters = None, 0
    if lambda_soft_contingency > 0.0:
        logger.info("Computing Leiden over-clustering for soft contingency loss...")
        labels_np, _ = run_leiden_clustering(adata_sc, resolution=leiden_resolution)
        leiden_n_clusters = int(labels_np.max()) + 1
        leiden_labels_tensor = torch.tensor(labels_np, dtype=torch.long, device=device)
        logger.info(f"Computed {leiden_n_clusters} clusters.")

    # (Optional) Preprocess data: Normalize & Log-transform
    if normalize_and_log:
        logger.info("Normalize & Log-transform gene expression and spatial data")
        sc.pp.normalize_total(adata_sc)
        sc.pp.normalize_total(adata_st)
        sc.pp.log1p(adata_sc)
        sc.pp.log1p(adata_st)
    else:
        logger.info("Skipping normalize_and_log")

    # Compute a priori sc embedding Y (C x d)
    # logger.info(f"Computing sc embedding Y (method={sc_embedding_method}, d={sc_embedding_d})")
    # Y = compute_sc_embedding(
    #     adata_sc,
    #     method=sc_embedding_method,
    #     d=sc_embedding_d,
    #     device=device,
    #     cache_dir=sc_data_dir,
    # )
    # logger.info(f"Y shape: {Y.shape}")
    # Y_scale = Y.var(dim=0, unbiased=False).mean().item()
    # logger.info(f"Y_scale (mean per-dim variance, used to normalise clust_intra loss): {Y_scale:.4f}")

    # Convert input anndata to tensors
    logger.debug("Prepare input tensors for model...")
    X, Z, X_shared, Z_shared = prepare_tensors_from_input(adata_sc, adata_st, device)
    logger.info("Prepared input tensors for model.")
    logger.info(f"Shared genes between scRNA and ST: {X_shared.shape[1]}")

    # Initialize Model
    num_spots, g_st = Z.shape
    num_cells, g_sc = X.shape
    num_genes_shared = X_shared.shape[1]
    logger.debug(f"ST dimensions: num_spots={num_spots}, g_st={g_st}")
    logger.debug(f"scRNA dimensions: num_cells={num_cells}, g_sc={g_sc}")
    logger.debug(f"Number of genes shared: {num_genes_shared}")

    # GPU memory guard
    estimated_gb = estimate_gpu_memory_gb(
        num_cells=num_cells,
        num_spots=num_spots,
        g_sc=g_sc,
        g_st=g_st,
        num_genes_shared=num_genes_shared,
        K=K,
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
        num_cells_sc=num_cells,
        k=K,
    ).to(device)

    # Initialize Loss and Optimizer
    loss = AIMLoss(
        lambda_rec_spot=lambda_rec_spot,
        lambda_rec_gene=lambda_rec_gene,
        # lambda_clust_intra=lambda_clust_intra,
        # lambda_clust_inter=lambda_clust_inter,
        lambda_state_entropy=lambda_state_entropy,
        lambda_spot_entropy=lambda_spot_entropy,
        lambda_soft_contingency=lambda_soft_contingency,
        k=K,
        leiden_labels=leiden_labels_tensor,
        leiden_n_clusters=leiden_n_clusters,
        # Y_scale=Y_scale,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 8. Training Loop
    logger.info("Starting optimization loop")
    model.train()

    # Collect per-epoch losses for plotting: individual components + total
    losses: dict[str, Any] = {
        "total-weighted": list(),
        "rec_spot": {"weight": lambda_rec_spot, "values": list()},
        "rec_gene": {"weight": lambda_rec_gene, "values": list()},
        # "clust_intra": {"weight": lambda_clust_intra, "values": list()},
        # "clust_inter": {"weight": lambda_clust_inter, "values": list()},
        "state_entropy": {"weight": lambda_state_entropy, "values": list()},
        "spot_entropy": {"weight": lambda_spot_entropy, "values": list()},
        **(
            {"soft_contingency": {"weight": lambda_soft_contingency, "values": list()}}
            if lambda_soft_contingency > 0.0
            else {}
        ),
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
        A, B = model()

        # Calculate segmented losses
        loss_dict = loss(
            A=A,
            B=B,
            X_shared=X_shared,
            Z_shared=Z_shared,
            # Y=Y
        )
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
        # losses["clust_intra"]["values"].append(to_scalar(loss_dict.get("clust_intra")))
        # losses["clust_inter"]["values"].append(to_scalar(loss_dict.get("clust_inter")))
        losses["state_entropy"]["values"].append(
            to_scalar(loss_dict.get("state_entropy"))
        )
        losses["spot_entropy"]["values"].append(
            to_scalar(loss_dict.get("spot_entropy"))
        )
        if lambda_soft_contingency > 0.0:
            losses["soft_contingency"]["values"].append(
                to_scalar(loss_dict.get("soft_contingency"))
            )
        # Logging: verbose -> log every epoch at DEBUG, normal -> log every 10 epochs at INFO
        if verbose_logging:
            logger.debug(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")
        else:
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")

    if save_intermediate:
        logger.info("Saving intermediate results...")
        folder_intermediate = output_folder / "intermediate"
        folder_intermediate.mkdir(parents=True, exist_ok=True)

        C = torch.matmul(A, B)

        K = B.shape[1]
        state_names = [f"state_{i}" for i in range(K)]

        B_thresh = B.detach().cpu().clone()
        B_thresh[B_thresh < 0.1] = 0.0
        arr_to_h5ad(
            B_thresh.numpy(),
            folder_intermediate / "B_thresh.h5ad",
            obs_names=adata_sc.obs_names.tolist(),
            var_names=state_names,
        )

        C_thresh = C.detach().cpu().clone()
        C_thresh[C_thresh < 0.1] = 0.0
        arr_to_h5ad(
            C_thresh.numpy(),
            folder_intermediate / "C_thresh.h5ad",
            obs_names=adata_st.obs_names.tolist(),
            var_names=state_names,
        )

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

        assert set(state_usage["C_used_indices"]).issubset(
            set(state_usage["B_used_indices"])
        ), (
            f"C_used_indices is not a subset of B_used_indices: "
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

        q = torch.mean(C, dim=0)
        q_np = q.detach().cpu().numpy()
        pd.DataFrame(q_np.round(2)).to_csv(folder_intermediate / "q.csv", index=False)
        arr_to_h5ad(
            q_np.reshape(1, -1),
            folder_intermediate / "q.h5ad",
            obs_names=["mean_spot_state_usage"],
            var_names=state_names,
        )

        B_normalized = B / (torch.sum(B, dim=0) + 1e-6)

        # M = torch.matmul(B.t(), X)
        # df = pd.DataFrame(M.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "M.csv", index=False)

        M_normalized = torch.matmul(B_normalized.t(), X)
        M_np = M_normalized.detach().cpu().numpy()
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

        # Z_prime = torch.matmul(A, X)
        # df = pd.DataFrame(Z_prime.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "Z_prime.csv", index=False)

        # CM_normalized = torch.matmul(C, M_normalized)
        # df = pd.DataFrame(CM_normalized.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "CM_normalized.csv", index=False)

        # B: Argmax per cell -> One-hot pro Zeile
        argmax_idx = torch.argmax(B, dim=1, keepdim=True)
        B_argmax = torch.zeros_like(B)
        B_argmax.scatter_(1, argmax_idx, 1.0)
        B_argmax_np = B_argmax.detach().cpu().numpy()
        arr_to_h5ad(
            B_argmax_np,
            folder_intermediate / "B_argmax.h5ad",
            obs_names=adata_sc.obs_names.tolist(),
            var_names=state_names,
        )

        # C: Argmax per spot -> One-hot pro Zeile
        C_argmax = torch.matmul(A, B_argmax)
        argmax_idx = torch.argmax(C_argmax, dim=1, keepdim=True)
        C_argmax = torch.zeros_like(C_argmax)
        C_argmax.scatter_(1, argmax_idx, 1.0)
        C_argmax_np = C_argmax.detach().cpu().numpy()
        arr_to_h5ad(
            C_argmax_np,
            folder_intermediate / "C_argmax.h5ad",
            obs_names=adata_st.obs_names.tolist(),
            var_names=state_names,
        )

    logger.info("Alignment complete.")
    return A, B, losses


def compute_gene_expression_prediction(
    spot_to_cell_map: torch.Tensor,
    cell_to_cell_type: torch.Tensor,
    adata_sc: AnnData,
    adata_st: AnnData,
    deterministic_mapping: bool,
    torch_device: torch.device,
) -> AnnData:
    """
    Predict spot-level gene expression from the learned mapping.

    Computes Z' = C @ M where C = A @ B (spot-to-state) and
    M = (B / colsum(B))^T @ X (state gene expression profiles).

    In deterministic mode, B is replaced by its row-wise argmax (one-hot)
    before computing C, so each spot maps to exactly one state.

    Args:
        spot_to_cell_map: A — soft spot-to-cell matrix (S × C), rows sum to 1.
        cell_to_cell_type: B — soft cell-to-state matrix (C × K), rows sum to 1.
        adata_sc: scRNA-seq reference (C × G_sc).
        adata_st: Spatial transcriptomics data (S × G_st), used for spot IDs only.
        deterministic_mapping: If True, apply argmax before computing C.
        torch_device: Device for intermediate tensor computation.

    Returns:
        AnnData of shape (G_sc × S): predicted expression matrix,
        obs_names = gene symbols, var_names = spot IDs.
    """

    X_sc = adata_sc.X.toarray() if sp.issparse(adata_sc.X) else np.asarray(adata_sc.X)
    adata_sc_tensor = torch.as_tensor(X_sc, dtype=torch.float32, device=torch_device)

    if deterministic_mapping:

        # A: Leave as it is

        # B: Argmax per cell -> One-hot pro Zeile
        argmax_idx = torch.argmax(cell_to_cell_type, dim=1, keepdim=True)
        cell_to_cell_type = torch.zeros_like(cell_to_cell_type)
        cell_to_cell_type.scatter_(1, argmax_idx, 1.0)

        # C
        C = torch.matmul(spot_to_cell_map, cell_to_cell_type)  # S x T
        argmax_idx = torch.argmax(C, dim=1, keepdim=True)
        C_argmax = torch.zeros_like(C)
        C_argmax.scatter_(1, argmax_idx, 1.0)

    else:
        # A, B leave is they are
        C = torch.matmul(spot_to_cell_map, cell_to_cell_type)  # S x T

    # Compute M = B_normalized^T @ X, then Z' = C @ M
    B_normalized = cell_to_cell_type / (
        torch.sum(cell_to_cell_type, dim=0) + 1e-6
    )  # (K x C)
    M = torch.matmul(B_normalized.t(), adata_sc_tensor)  # T x G
    predicted_spot_expressions = torch.matmul(C, M)  # S x G

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
    K: int = 20,
    lr: float = 0.008,
    epochs: int = 1000,
    normalize_and_log: bool = False,
    leiden_resolution: float = 3.0,
    # sc_embedding_method: str = "pca",
    # sc_embedding_d: int = 32,
    lambda_rec_spot: float = 0.5,
    lambda_rec_gene: float = 0.5,
    # lambda_clust_intra: float = 0.0,
    # lambda_clust_inter: float = 0.0,
    lambda_state_entropy: float = 0.1,
    lambda_spot_entropy: float = 0.08,
    lambda_soft_contingency: float = 1.0,
    store_intermediate: bool = False,
    skip_analysis: bool = False,
    verbose_logging: bool = False,
    gpu_limit_gb: int = 6,
) -> tuple[AnnData, AnnData, AnnData, dict]:
    """
    Load data, run mapping, compute both probabilistic and deterministic GEPs,
    and write all outputs to output_folder.

    Outputs written:
        gep_prob.h5ad      — probabilistic predicted GEP (G × S)
        gep_det.h5ad       — deterministic predicted GEP (G × S)
        mapping_prob.h5ad  — soft spot-to-cell map (C × S)
        mapping_det.h5ad   — hard spot-to-cell map (C × S, one-hot columns)
        loss/              — per-epoch loss curves and final values CSV
        intermediate/      — B, C, M matrices (only when store_intermediate=True)

    Returns:
        gep_prob:               Probabilistic GEP AnnData (G × S).
        gep_det:                Deterministic GEP AnnData (G × S).
        cell_to_cell_type_adata: Cell-to-state assignment matrix as AnnData (C × K).
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
    spot_to_cell_map, cell_to_cell_type, losses = aim_compute_mapping(
        adata_sc=adata_sc.copy(),
        adata_st=adata_st.copy(),
        output_folder=output_folder,
        K=K,
        lr=lr,
        epochs=epochs,
        normalize_and_log=normalize_and_log,
        leiden_resolution=leiden_resolution,
        # sc_embedding_method=sc_embedding_method,
        # sc_embedding_d=sc_embedding_d,
        lambda_rec_spot=lambda_rec_spot,
        lambda_rec_gene=lambda_rec_gene,
        # lambda_clust_intra=lambda_clust_intra,
        # lambda_clust_inter=lambda_clust_inter,
        lambda_state_entropy=lambda_state_entropy,
        lambda_spot_entropy=lambda_spot_entropy,
        lambda_soft_contingency=lambda_soft_contingency,
        verbose_logging=verbose_logging,
        device=device,
        save_intermediate=store_intermediate,
        gpu_limit_gb=gpu_limit_gb,
        # sc_data_dir=sc_path.parent,
    )
    logger.info("Obtained spot-to-cell mapping.")
    assert spot_to_cell_map.shape == (
        adata_st.n_obs,
        adata_sc.n_obs,
    ), "dims passen nicht"

    # Create adata containing clustering information (cell to cell type)
    cell_to_cell_type_adata = AnnData(X=cell_to_cell_type.detach().cpu().numpy())
    cell_to_cell_type_adata.obs_names = adata_sc.obs_names
    cell_to_cell_type_adata.var_names = [
        f"Type {n}" for n in range(cell_to_cell_type.shape[1])
    ]

    # Save loss logs and plots
    losses_after_last_epoch = dump_loss_logs(losses, output_folder)
    create_loss_plots(losses, output_folder / "loss")

    # Compute and save probabilistic GEP
    adata_prediction_prob = compute_gene_expression_prediction(
        spot_to_cell_map,
        cell_to_cell_type,
        adata_sc,
        adata_st,
        False,
        device,
    )
    gep_prob_path = output_folder / "gep_prob.h5ad"
    adata_prediction_prob.write_h5ad(gep_prob_path)
    logger.info(f"Saved prob GEP to {gep_prob_path}")

    mapping_np = spot_to_cell_map.detach().cpu().numpy().T  # C×S
    mapping_prob_path = output_folder / "mapping_prob.h5ad"
    AnnData(
        X=mapping_np.astype(np.float32),
        obs=pd.DataFrame(index=adata_sc.obs_names),
        var=pd.DataFrame(index=adata_st.obs_names),
    ).write_h5ad(mapping_prob_path)
    logger.info(f"Saved prob mapping to {mapping_prob_path}")

    # Compute and save deterministic GEP
    logger.info("Apply deterministic mapping & compute prediction")
    adata_prediction_det = compute_gene_expression_prediction(
        spot_to_cell_map,
        cell_to_cell_type,
        adata_sc,
        adata_st,
        True,
        device,
    )
    gep_det_path = output_folder / "gep_det.h5ad"
    adata_prediction_det.write_h5ad(gep_det_path)
    logger.info(f"Saved det GEP to {gep_det_path}")

    argmax_idx = np.argmax(mapping_np, axis=0)
    one_hot = np.zeros_like(mapping_np, dtype=np.uint8)
    one_hot[argmax_idx, np.arange(mapping_np.shape[1])] = 1
    mapping_det_path = output_folder / "mapping_det.h5ad"
    AnnData(
        X=one_hot,
        obs=pd.DataFrame(index=adata_sc.obs_names),
        var=pd.DataFrame(index=adata_st.obs_names),
    ).write_h5ad(mapping_det_path)
    logger.info(f"Saved det mapping to {mapping_det_path}")

    # Post-mapping analysis and report
    if not skip_analysis:
        try:
            B_np = cell_to_cell_type.detach().cpu().numpy()
            C_np = spot_to_cell_map.detach().cpu().numpy() @ B_np  # S×K
            B_np[B_np < 0.1] = 0.0
            C_np[C_np < 0.1] = 0.0
            analysis_dir = output_folder / "analysis"
            run_analysis(
                adata_sc=adata_sc,
                adata_st=adata_st,
                B=B_np,
                C=C_np,
                output_dir=analysis_dir,
                K=K,
                leiden_resolution=leiden_resolution,
            )
            generate_per_k_report(analysis_dir, K, run_id="0")
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
    # model
    parser.add_argument(
        "--K", type=int, default=20, help="Number of cell-type clusters (model.K)"
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
        help="Leiden clustering resolution for soft-contingency loss (training.reference_leiden_clustering_resolution)",
    )
    # sc_embedding
    # parser.add_argument(
    #     "--sc_embedding_method",
    #     choices=["pca", "scvi"],
    #     default="pca",
    #     help="SC embedding method (sc_embedding.method)",
    # )
    # parser.add_argument(
    #     "--sc_embedding_d",
    #     type=int,
    #     default=32,
    #     help="SC embedding dimensionality (sc_embedding.d)",
    # )
    # loss weights
    parser.add_argument("--lambda_rec_spot", type=float, default=0.5)
    parser.add_argument("--lambda_rec_gene", type=float, default=0.5)
    # parser.add_argument("--lambda_clust_intra", type=float, default=0.0)
    # parser.add_argument("--lambda_clust_inter", type=float, default=0.0)
    parser.add_argument("--lambda_state_entropy", type=float, default=0.1)
    parser.add_argument("--lambda_spot_entropy", type=float, default=0.08)
    parser.add_argument("--lambda_soft_contingency", type=float, default=1.0)
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
                "model": {"K": args.K},
                "training": {
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "normalize_and_log": args.normalize_and_log,
                    "reference_leiden_clustering_resolution": args.leiden_resolution,
                },
                # "sc_embedding": {
                #     "method": args.sc_embedding_method,
                #     "d": args.sc_embedding_d,
                # },
                "loss_weights": {
                    "lambda_rec_spot": args.lambda_rec_spot,
                    "lambda_rec_gene": args.lambda_rec_gene,
                    # "lambda_clust_intra": args.lambda_clust_intra,
                    # "lambda_clust_inter": args.lambda_clust_inter,
                    "lambda_state_entropy": args.lambda_state_entropy,
                    "lambda_spot_entropy": args.lambda_spot_entropy,
                    "lambda_soft_contingency": args.lambda_soft_contingency,
                },
            },
            f,
            sort_keys=False,
        )

    main(
        args.scdata,
        args.stdata,
        output_folder=args.output_folder,
        K=args.K,
        lr=args.lr,
        epochs=args.epochs,
        normalize_and_log=args.normalize_and_log,
        leiden_resolution=args.leiden_resolution,
        # sc_embedding_method=args.sc_embedding_method,
        # sc_embedding_d=args.sc_embedding_d,
        lambda_rec_spot=args.lambda_rec_spot,
        lambda_rec_gene=args.lambda_rec_gene,
        # lambda_clust_intra=args.lambda_clust_intra,
        # lambda_clust_inter=args.lambda_clust_inter,
        lambda_state_entropy=args.lambda_state_entropy,
        lambda_spot_entropy=args.lambda_spot_entropy,
        lambda_soft_contingency=args.lambda_soft_contingency,
        verbose_logging=(args.logging == "verbose"),
        store_intermediate=True,
        gpu_limit_gb=args.gpu_limit_gb,
    )
