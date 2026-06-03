import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import yaml
import scanpy as sc
import torch
import torch.optim as optim
import logging
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from ..utils.io import anndata_to_csv, load_sc_adata, load_st_adata
from .src.utils import (
    fmt_nonzero_4,
    create_loss_plots,
    dump_loss_logs,
    build_sc_knn_graph,
)
from .src.model import AlternativeIdeaModel
from .src.loss import AlternativeIdeaLoss
from .src.spatial_graph import build_spatial_graph, SpatialGraphType
from .src.dataset import prepare_tensors_from_input
from .src.evaluate_k.clustering import run_leiden_clustering
from .src.sc_embedding import compute_sc_embedding

logger = logging.getLogger(__name__)

GPU_LIMIT_GB = 6.0


def estimate_gpu_memory_gb(
    num_cells: int,
    num_spots: int,
    g_sc: int,
    g_st: int,
    num_genes_shared: int,
    K: int,
) -> float:
    """Rough upper-bound estimate of GPU memory required (in GB)."""
    B32 = 4  # bytes per float32

    # Input tensors kept on GPU during training
    data = (
        num_cells * g_sc
        + num_spots * g_st
        + num_cells * num_genes_shared
        + num_spots * num_genes_shared
    ) * B32

    # Trainable parameters + their gradients (factor ×2) + Adam moment estimates (factor ×2)
    A_bytes = num_spots * num_cells * B32 * 4
    B_bytes = num_cells * K * B32 * 4

    # 50 % overhead for activations, edge index, misc buffers
    total = (data + A_bytes + B_bytes) * 1.5
    return total / (1024**3)


def _arr_to_h5ad(
    arr: np.ndarray,
    path: Path,
    obs_names: list,
    var_names: list,
) -> None:
    """Save a 2D numpy array as an h5ad file with labelled obs and var axes."""
    import numpy as np

    adata = AnnData(X=arr.astype(np.float32))
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.write_h5ad(path)


def load_config(config_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    if not os.path.exists(config_path):
        raise Exception(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:

        # Load yaml config
        cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Top-level config must be a mapping (dict).")

        # Ensure required sections exist
        required_sections = [
            "mapping",
            "model",
            "training",
            "loss_weights",
            "sc_embedding",
        ]
        missing_sections = [s for s in required_sections if s not in cfg]
        if missing_sections:
            raise ValueError(
                f"Missing required config sections: {', '.join(missing_sections)}"
            )

        mapping_cfg = cfg.get("mapping") if isinstance(cfg.get("mapping"), dict) else {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        training_cfg = (
            cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
        )
        loss_weights_cfg = (
            cfg.get("loss_weights") if isinstance(cfg.get("loss_weights"), dict) else {}
        )
        sc_embedding_cfg = (
            cfg.get("sc_embedding") if isinstance(cfg.get("sc_embedding"), dict) else {}
        )

        # Validate mapping config
        if not mapping_cfg:
            raise ValueError("`mapping` must be a mapping in the config.")
        for key in ("deterministic",):
            if key not in mapping_cfg:
                raise ValueError(f"`mapping.{key}` is required in the config.")

        # Validate model config
        if not model_cfg:
            raise ValueError("`model` must be a mapping in the config.")
        for key in ("K",):
            if key not in model_cfg:
                raise ValueError(f"`model.{key}` is required in the config.")

        # Validate training config
        if not training_cfg:
            raise ValueError("`training` must be a mapping in the config.")
        for key in ("lr", "epochs", "use_cm"):
            if key not in training_cfg:
                raise ValueError(f"`training.{key}` is required in the config.")
        # Optional normalize_and_log flag: default True (preserve previous behavior)
        if "normalize_and_log" in training_cfg:
            if not isinstance(training_cfg["normalize_and_log"], bool):
                raise ValueError(
                    "`training.normalize_and_log` must be a boolean if provided."
                )
        else:
            training_cfg["normalize_and_log"] = True

        # Validate loss_weights is a mapping (specific keys may be optional)
        if not isinstance(loss_weights_cfg, dict):
            raise ValueError("`loss_weights` must be a mapping in the config.")

        # Validate sc_embedding config
        if not isinstance(sc_embedding_cfg, dict):
            raise ValueError("`sc_embedding` must be a mapping in the config.")
        sc_emb_method = sc_embedding_cfg.get("method")
        if not isinstance(sc_emb_method, str) or sc_emb_method not in ("pca", "scvi"):
            raise ValueError("`sc_embedding.method` must be 'pca' or 'scvi'.")
        sc_emb_d = sc_embedding_cfg.get("d")
        if not isinstance(sc_emb_d, int) or sc_emb_d <= 0:
            raise ValueError("`sc_embedding.d` must be a positive integer.")

    return (
        mapping_cfg,
        model_cfg,
        training_cfg,
        loss_weights_cfg,
        sc_embedding_cfg,
    )


def alternative_idea_compute_mapping(
    path_to_config: Path,
    adata_sc: AnnData,
    adata_st: AnnData,
    verbose_logging: bool,
    device: torch.device,
    save_intermediate: bool = False,
    no_gpu_limit: bool = False,
    sc_data_dir: Optional[Path] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:

    # 1. Load Config
    logger.debug(f"Load config: {path_to_config}")
    (
        mapping_config,
        model_config,
        training_config,
        loss_weights,
        sc_embedding_config,
    ) = load_config(path_to_config)
    logger.info(f"Loaded config: {path_to_config}")
    logger.debug(f"Loaded training config: {training_config}")
    logger.debug(f"Loaded model config: {model_config}")
    logger.debug(f"Loaded loss weights: {loss_weights}")
    logger.info(f"Using device: {device}")

    # 3. Leiden over-clustering for soft contingency loss (once, before training)
    # Must happen before prepare_tensors_from_input so adata_sc is still available
    leiden_labels, leiden_n_clusters = None, 0
    if loss_weights.get("lambda_soft_contingency", 0.0) > 0.0:
        leiden_resolution = training_config.get(
            "reference_leiden_clustering_resolution"
        )
        logger.info("Computing Leiden over-clustering for soft contingency loss...")
        assert leiden_resolution is not None and type(leiden_resolution) == float
        labels_np, _ = run_leiden_clustering(adata_sc, resolution=leiden_resolution)
        leiden_n_clusters = int(labels_np.max()) + 1
        leiden_labels = torch.tensor(labels_np, dtype=torch.long, device=device)
        logger.info(f"Computed {leiden_n_clusters} clusters.")

    # (Optional) Preprocess data (only for mapping): Normalize & Log-transform
    if training_config["normalize_and_log"]:
        logger.info("Normalize & Log-transform gene expression and spatial data")
        sc.pp.normalize_total(adata_sc)
        sc.pp.normalize_total(adata_st)
        sc.pp.log1p(adata_sc)
        sc.pp.log1p(adata_st)
    else:
        logger.info(
            "Skipping normalize_and_log as per config (training.normalize_and_log=false)"
        )

    # 4. Compute a priori sc embedding Y (C x d)
    logger.info(
        f"Computing sc embedding Y (method={sc_embedding_config['method']}, d={sc_embedding_config['d']})"
    )
    Y = compute_sc_embedding(
        adata_sc,
        method=sc_embedding_config["method"],
        d=sc_embedding_config["d"],
        device=device,
        cache_dir=sc_data_dir,
    )
    logger.info(f"Y shape: {Y.shape}")
    Y_scale = Y.var(dim=0, unbiased=False).mean().item()
    logger.info(
        f"Y_scale (mean per-dim variance, used to normalise clust_intra loss): {Y_scale:.4f}"
    )

    # 5. Convert input anndata to tensors
    logger.debug("Prepare input tensors for model...")
    X, Z, X_shared, Z_shared = prepare_tensors_from_input(adata_sc, adata_st, device)
    logger.info("Prepared input tensors for model.")
    logger.info(f"Shared genes between scRNA and ST: {X_shared.shape[1]}")

    # 5. Build the spatial graph out of Z
    edge_index = build_spatial_graph(
        adata_st,
        method=SpatialGraphType.KNN,
        k=4,
        radius=None,
        device=device,
    )
    logger.info("Created spatial graph (KNN, k=4).")

    # 6. Initialize Model
    # Dimensions derived from tensors
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
        K=model_config["K"],
    )
    logger.info(f"Estimated GPU memory requirement: {estimated_gb:.2f} GB")
    if not no_gpu_limit and estimated_gb > GPU_LIMIT_GB:
        logger.error(
            f"Estimated GPU memory ({estimated_gb:.2f} GB) exceeds the {GPU_LIMIT_GB:.0f} GB limit. "
            f"Aborting to prevent OOM. Pass --no_gpu_limit to bypass this check."
        )
        sys.exit(1)
    if no_gpu_limit and estimated_gb > 128.0:
        logger.error(
            f"Estimated GPU memory ({estimated_gb:.2f} GB) exceeds the 128GB limit. "
            f"Aborting to prevent OOM. Pass --no_gpu_limit to bypass this check."
        )
        sys.exit(1)

    model = AlternativeIdeaModel(
        num_spots_st=num_spots,
        num_cells_sc=num_cells,
        k=model_config["K"],
    ).to(device)

    # 7. Initialize Loss and Optimizer

    # Precompute sc KNN graph for soft modularity loss (once, before training starts)
    knn_W, knn_k, knn_two_m = None, None, 1.0
    if loss_weights.get("lambda_soft_modularity", 0.0) > 0.0:
        logger.info("Precomputing sc KNN graph for soft modularity loss...")
        knn_W, knn_k, knn_two_m = build_sc_knn_graph(
            X, n_pca_components=50, n_neighbors=15, device=device
        )

    loss = AlternativeIdeaLoss(
        lambda_rec_spot=loss_weights["lambda_rec_spot"],
        lambda_rec_gene=loss_weights["lambda_rec_gene"],
        lambda_clust_intra=loss_weights.get("lambda_clust_intra", 0.0),
        lambda_clust_inter=loss_weights.get("lambda_clust_inter", 0.0),
        lambda_state_entropy=loss_weights["lambda_state_entropy"],
        lambda_spot_entropy=loss_weights["lambda_spot_entropy"],
        lambda_soft_modularity=loss_weights.get("lambda_soft_modularity", 0.0),
        lambda_soft_contingency=loss_weights.get("lambda_soft_contingency", 0.0),
        k=model_config["K"],
        use_cm=bool(training_config["use_cm"]),
        knn_W=knn_W,
        knn_k=knn_k,
        knn_two_m=knn_two_m,
        leiden_labels=leiden_labels,
        leiden_n_clusters=leiden_n_clusters,
        Y_scale=Y_scale,
    )

    optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])

    # 8. Training Loop
    logger.info("Starting optimization loop")
    model.train()

    # Collect per-epoch losses for plotting: individual components + total
    losses: dict[str, Any] = {
        "total-weighted": list(),
        "rec_spot": {"weight": loss_weights["lambda_rec_spot"], "values": list()},
        "rec_gene": {"weight": loss_weights["lambda_rec_gene"], "values": list()},
        "clust_intra": {
            "weight": loss_weights.get("lambda_clust_intra", 0.0),
            "values": list(),
        },
        "clust_inter": {
            "weight": loss_weights.get("lambda_clust_inter", 0.0),
            "values": list(),
        },
        "state_entropy": {
            "weight": loss_weights["lambda_state_entropy"],
            "values": list(),
        },
        "spot_entropy": {
            "weight": loss_weights["lambda_spot_entropy"],
            "values": list(),
        },
        **(
            {
                "soft_modularity": {
                    "weight": loss_weights.get("lambda_soft_modularity", 0.0),
                    "values": list(),
                }
            }
            if loss_weights.get("lambda_soft_modularity", 0.0) > 0.0
            else {}
        ),
        **(
            {
                "soft_contingency": {
                    "weight": loss_weights.get("lambda_soft_contingency", 0.0),
                    "values": list(),
                }
            }
            if loss_weights.get("lambda_soft_contingency", 0.0) > 0.0
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

    # Initialize A so we can safely return it even when epochs == 0
    A = None
    B = None

    for epoch in range(int(training_config["epochs"])):
        optimizer.zero_grad()

        # Forward pass
        A, B, h, M_rec, F = model(Z, edge_index)

        # Calculate segmented losses
        loss_dict = loss(A=A, B=B, X_shared=X_shared, Z_shared=Z_shared, Y=Y)
        total_loss = loss_dict["loss"]

        # 3. Optionaler Gradient-Check (Nur zur Diagnose)
        if epoch % 100 == 0 and verbose_logging:
            logger.debug(f"\n--- Gradient Analysis (Epoch {epoch}) ---")
            for name, loss_val in loss_dict.items():
                if name == "loss":
                    continue

                # Temporäre Gradientenberechnung für diesen spezifischen Term
                model.zero_grad()
                loss_val.backward(retain_graph=True)

                # Berechne die Norm über alle trainierbaren Gewichte
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1000
                )
                logger.debug(
                    f"Term: {name:15} | Loss: {loss_val.item():.4f} | Grad-Norm: {grad_norm:.4f}"
                )

            # Reset für den eigentlichen Backprop
            model.zero_grad()

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Collect loss values
        losses["total-weighted"].append(to_scalar(total_loss))
        losses["rec_spot"]["values"].append(to_scalar(loss_dict.get("rec_spot")))
        losses["rec_gene"]["values"].append(to_scalar(loss_dict.get("rec_gene")))
        losses["clust_intra"]["values"].append(to_scalar(loss_dict.get("clust_intra")))
        losses["clust_inter"]["values"].append(to_scalar(loss_dict.get("clust_inter")))
        losses["state_entropy"]["values"].append(
            to_scalar(loss_dict.get("state_entropy"))
        )
        losses["spot_entropy"]["values"].append(
            to_scalar(loss_dict.get("spot_entropy"))
        )
        if loss_weights.get("lambda_soft_modularity", 0.0) > 0.0:
            losses["soft_modularity"]["values"].append(
                to_scalar(loss_dict.get("soft_modularity"))
            )
        if loss_weights.get("lambda_soft_contingency", 0.0) > 0.0:
            losses["soft_contingency"]["values"].append(
                to_scalar(loss_dict.get("soft_contingency"))
            )
        # Logging: verbose -> log every epoch at DEBUG, normal -> log every 10 epochs at INFO
        if verbose_logging:
            logger.debug(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")
        else:
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")

    assert A is not None and B is not None

    if save_intermediate:
        logger.info("Saving intermediate results...")
        folder_intermediate = path_to_config.parent / "intermediate"
        folder_intermediate.mkdir(parents=True, exist_ok=True)

        # df = pd.DataFrame(X.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "X.csv", index=False)

        # df = pd.DataFrame(Z.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "Z.csv", index=False)

        # df = pd.DataFrame(X_shared.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "X_shared.csv", index=False)

        # df = pd.DataFrame(Z_shared.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "Z_shared.csv", index=False)

        # df = pd.DataFrame(A.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "A.csv", index=False)

        # df = pd.DataFrame(B.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "B.csv", index=False)

        C = torch.matmul(A, B)
        # df = pd.DataFrame(C.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "C.csv", index=False)

        K = B.shape[1]
        state_names = [f"state_{i}" for i in range(K)]

        # A_thresh = A.detach().cpu().clone()
        # A_thresh[A_thresh < 0.1] = 0.0
        # df_weight = pd.DataFrame(A_thresh.numpy())
        # df_weight.to_csv(folder_intermediate / "A_thresh.csv", index=False)
        # _arr_to_h5ad(
        #     A_thresh.numpy(),
        #     folder_intermediate / "A_thresh.h5ad",
        #     obs_names=adata_st.obs_names.tolist(),
        #     var_names=adata_sc.obs_names.tolist(),
        # )

        B_thresh = B.detach().cpu().clone()
        B_thresh[B_thresh < 0.1] = 0.0
        # df_weight = pd.DataFrame(B_thresh.numpy())
        # df_weight.to_csv(folder_intermediate / "B_thresh.csv", index=False)
        _arr_to_h5ad(
            B_thresh.numpy(),
            folder_intermediate / "B_thresh.h5ad",
            obs_names=adata_sc.obs_names.tolist(),
            var_names=state_names,
        )

        C_thresh = C.detach().cpu().clone()
        C_thresh[C_thresh < 0.1] = 0.0
        # df_weight = pd.DataFrame(C_thresh.numpy())
        # df_weight.to_csv(folder_intermediate / "C_thresh.csv", index=False)
        _arr_to_h5ad(
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

        # idx = torch.argmax(A, dim=1, keepdim=True)
        # A_argmax = torch.zeros_like(A).scatter_(1, idx, 1.0)
        # pd.DataFrame(A_argmax.detach().cpu().numpy()).to_csv(folder_intermediate / "A_argmax.csv", index=False)

        # idx = torch.argmax(B, dim=1, keepdim=True)
        # B_argmax = torch.zeros_like(B).scatter_(1, idx, 1.0)
        # pd.DataFrame(B_argmax.detach().cpu().numpy()).to_csv(folder_intermediate / "B_argmax.csv", index=False)

        # idx = torch.argmax(C, dim=1, keepdim=True)
        # C_argmax = torch.zeros_like(C).scatter_(1, idx, 1.0)
        # pd.DataFrame(C_argmax.detach().cpu().numpy()).to_csv(folder_intermediate / "C_argmax.csv", index=False)

        p = torch.mean(B, dim=0)
        p_np = p.detach().cpu().numpy()
        pd.DataFrame(p_np.round(2)).to_csv(folder_intermediate / "p.csv", index=False)
        _arr_to_h5ad(
            p_np.reshape(1, -1),
            folder_intermediate / "p.h5ad",
            obs_names=["mean_cell_state_usage"],
            var_names=state_names,
        )

        q = torch.mean(C, dim=0)
        q_np = q.detach().cpu().numpy()
        pd.DataFrame(q_np.round(2)).to_csv(folder_intermediate / "q.csv", index=False)
        _arr_to_h5ad(
            q_np.reshape(1, -1),
            folder_intermediate / "q.h5ad",
            obs_names=["mean_spot_state_usage"],
            var_names=state_names,
        )

        B_normalized = B / (torch.sum(B, dim=0) + 1e-6)
        # df_weight = pd.DataFrame(B_normalized.detach().cpu().numpy())
        # df_weight.to_csv(folder_intermediate / 'B_normalized.csv', index=False)

        # M = torch.matmul(B.t(), X)
        # df = pd.DataFrame(M.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "M.csv", index=False)

        M_normalized = torch.matmul(B_normalized.t(), X)
        M_np = M_normalized.detach().cpu().numpy()
        # df = pd.DataFrame(M_np)
        # df.to_csv(folder_intermediate / "M_normalized.csv", index=False)
        _arr_to_h5ad(
            M_np,
            folder_intermediate / "M_normalized.h5ad",
            obs_names=state_names,
            var_names=adata_sc.var_names.tolist(),
        )

        c_used_idx = state_usage["C_used_indices"]
        cell_states = M_np[c_used_idx, :]
        # pd.DataFrame(cell_states).to_csv(
        #     folder_intermediate / "cell_states_gep.csv", index=False
        # )
        _arr_to_h5ad(
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

        if mapping_config.get("deterministic"):

            # B: Argmax per cell -> One-hot pro Zeile
            argmax_idx = torch.argmax(B, dim=1, keepdim=True)
            B_argmax = torch.zeros_like(B)
            B_argmax.scatter_(1, argmax_idx, 1.0)
            B_argmax_np = B_argmax.detach().cpu().numpy()
            # pd.DataFrame(B_argmax_np).to_csv(
            #     folder_intermediate / "B_argmax.csv", index=False
            # )
            _arr_to_h5ad(
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
            # pd.DataFrame(C_argmax_np).to_csv(
            #     folder_intermediate / "C_argmax.csv", index=False
            # )
            _arr_to_h5ad(
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
    use_cm: bool,
) -> AnnData:

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

    if use_cm:
        logger.info(
            "Using CM for final mapping output as per config (training.use_cm=true)"
        )
        # Compute M = B_normalized^T @ X
        B_normalized = cell_to_cell_type / (
            torch.sum(cell_to_cell_type, dim=0) + 1e-6
        )  # (K x C)
        M = torch.matmul(B_normalized.t(), adata_sc_tensor)  # T x G
        # Compute Z' = C @ M
        predicted_spot_expressions = torch.matmul(C, M)  # S x G
    else:
        predicted_spot_expressions = torch.matmul(
            spot_to_cell_map, adata_sc_tensor
        )  # S x G

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
    config_path: Path,
    output_path: Optional[Path],
    store_intermediate: bool = False,
    verbose_logging: bool = False,
    no_gpu_limit: bool = False,
) -> tuple[AnnData, Optional[AnnData], AnnData, dict]:

    mapping_config, _, training_config, _, _ = load_config(config_path)

    # Setup Device
    # Use 'mps' for Apple Silicon, 'cuda' for NVIDIA, or 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Step 1: Load data
    logger.info("Load input scRNA and ST data...")
    adata_sc = load_sc_adata(sc_path)  # C x G
    adata_st = load_st_adata(st_path)  # S x G
    logger.info("Loaded input scRNA and ST data.")

    # Step 2: Map data using AlternativeIdea
    spot_to_cell_map: torch.Tensor
    spot_to_cell_map, cell_to_cell_type, losses = alternative_idea_compute_mapping(
        config_path,
        adata_sc.copy(),
        adata_st.copy(),
        verbose_logging=verbose_logging,
        device=device,
        save_intermediate=store_intermediate,
        no_gpu_limit=no_gpu_limit,
        sc_data_dir=sc_path.parent,
    )  # S x C, plus loss history
    logger.info("Obtained spot-to-cell mapping AnnData.")
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

    # Step 3.1: Save end values of loss terms (value after last epoch) to a csv file
    losses_after_last_epoch = dump_loss_logs(losses, config_path)

    # Step 3.2: Create plots for loss curves
    create_loss_plots(losses, config_path.parent / "loss")

    # Step 4: Compute Z' out of the mapping (expected gene expression per spot, scRNA data weighted by mapping)
    adata_prediction_prob = compute_gene_expression_prediction(
        spot_to_cell_map,
        cell_to_cell_type,
        adata_sc,
        adata_st,
        False,
        device,
        training_config["use_cm"],
    )

    # Step 5 (optional): Export prob GEP + mapping to h5ad
    # Layout: obs = genes (G), var = spots (S)
    if output_path is not None:
        output_path = Path(output_path).with_suffix(".h5ad")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        adata_prediction_prob.write_h5ad(output_path)
        logger.info(f"Saved prob GEP to {output_path}")

        mapping_np = spot_to_cell_map.detach().cpu().numpy().T  # C×S
        mapping_prob_path = output_path.parent / "mapping_prob.h5ad"
        AnnData(
            X=mapping_np.astype(np.float32),
            obs=pd.DataFrame(index=adata_sc.obs_names),
            var=pd.DataFrame(index=adata_st.obs_names),
        ).write_h5ad(mapping_prob_path)
        logger.info(f"Saved prob mapping to {mapping_prob_path}")
    else:
        logger.debug("No output path provided, skipping h5ad export.")

    # Step 6 (optional): Apply one-hot encoding to mapping & repeat steps 4 & 5
    adata_prediction_det = None
    if mapping_config["deterministic"]:
        logger.info(
            "Apply deterministic mapping & compute prediction with one-hot encoded mapping"
        )

        adata_prediction_det = compute_gene_expression_prediction(
            spot_to_cell_map,
            cell_to_cell_type,
            adata_sc,
            adata_st,
            True,
            device,
            training_config["use_cm"],
        )

        if output_path is not None:
            output_path_deterministic = output_path.with_name(
                output_path.stem + "_deterministic" + output_path.suffix
            )

            adata_prediction_det.write_h5ad(output_path_deterministic)
            logger.info(f"Saved det GEP to {output_path_deterministic}")

            mapping_np = spot_to_cell_map.detach().cpu().numpy().T  # C×S
            argmax_idx = np.argmax(mapping_np, axis=0)
            one_hot = np.zeros_like(mapping_np, dtype=np.uint8)
            one_hot[argmax_idx, np.arange(mapping_np.shape[1])] = 1
            mapping_det_path = output_path.parent / "mapping_det.h5ad"
            AnnData(
                X=one_hot,
                obs=pd.DataFrame(index=adata_sc.obs_names),
                var=pd.DataFrame(index=adata_st.obs_names),
            ).write_h5ad(mapping_det_path)
            logger.info(f"Saved det mapping to {mapping_det_path}")
        else:
            logger.debug("No output path provided, skipping h5ad export.")

    # Step 7: Return result
    return (
        adata_prediction_prob,
        adata_prediction_det,
        cell_to_cell_type_adata,
        losses_after_last_epoch,
    )


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="Run AlternativeIdea alignment on a dataset folder"
    )
    parser.add_argument(
        "--scdata", dest="scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", dest="stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "-c", "--config", dest="config", type=Path, help="Path to config.yaml"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        type=Path,
        required=False,
        default=None,
        help="Path where to store result to",
    )
    parser.add_argument(
        "--logging",
        dest="logging",
        choices=["normal", "verbose"],
        default="normal",
        help="Logging verbosity. Use 'verbose' for more logs.",
    )
    parser.add_argument(
        "--no_gpu_limit",
        dest="no_gpu_limit",
        action="store_true",
        default=False,
        help=f"Bypass the {GPU_LIMIT_GB:.0f} GB GPU memory guard and run regardless of estimated memory usage.",
    )
    args = parser.parse_args()

    # Configure logging based on argument
    level = logging.DEBUG if args.logging == "verbose" else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(level)

    # Run alignment
    main(
        args.scdata,
        args.stdata,
        args.config,
        args.output_path,
        verbose_logging=(args.logging == "verbose"),
        store_intermediate=True,
        no_gpu_limit=args.no_gpu_limit,
    )
