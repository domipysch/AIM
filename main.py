"""
Agglomerative-K sweep — the novel method (single entry point).

An interpretable alternative to jointly learning a merge matrix G. Instead of
letting a loss decide how Leiden over-clusters combine, we:

    1. Leiden over-cluster the reference -> L subclusters (+ centroids, sizes).
    2. Build the full agglomeration tree ONCE with average-linkage on the
       shared-gene cosine distance between centroids (scipy.linkage). This *is*
       "merge the two closest, recompute, repeat" — scipy does the recomputation.
    3. For every K from L down to 1: cut the tree at K states, assemble the
       (size-weighted) merged state profiles M, and map every ST spot onto those
       states. The hard tree cut plays the role of the cluster->state map; only
       the spot->state map P varies with K.
    4. Emit a summary.csv and decision plots (reconstruction vs K, merge cost vs
       K, dendrogram) so the user can pick the K they want.

The spot->state mapping step is MODULAR (choose with --mapping):

    greedy  (default) — zero-parameter nearest-centroid classifier. Each spot is
                        assigned to the state whose (size-weighted) centroid is
                        most cosine-similar to it on the shared genes:
                            P[s] = one_hot( argmax_k cos(Z_shared[s], M[k]) )
                        No training; P is one-hot by construction, so the soft and
                        deterministic reconstructions coincide. This is the
                        baseline that shows what the merge tree alone buys you.
    learned           — a soft P learned by gradient descent (src/learn_mapping.py),
                        minimizing spot-wise + gene-wise cosine distance with a
                        quadratic spot_gini sharpener (optional warmup).

Each K's folder is written in the exact layout the post-mapping analysis expects
(mapping_prob.h5ad = P, leiden_merge_prob.h5ad = the hard cut G,
leiden_overclustering.h5ad, config.yaml), so the full analysis is just
analysis.run_from_output.analyze_run on each K folder (with --full_analysis).

Run (single pair):
    PYTHONPATH=src python main.py \
        --scdata <sc.h5ad> --stdata <st.h5ad> --output_folder <out> \
        [--mapping greedy|learned] \
        [--leiden_resolution 3.0] [--normalize_and_log] \
        [--k_min 1] [--k_max <L>] [--k_step 1] [--full_analysis] \
        # learned-mode only:
        [--epochs 400] [--lr 0.02] [--lambda_spot_gini 1.0] \
        [--spot_gini_warmup_frac 0.5]

Run (all pairs in pairs.csv, one after the other):
    PYTHONPATH=src python main.py \
        --pairs_csv <pairs.csv> --sc_dir <scRNA dir> --st_dir <ST dir> \
        --output_dir <out> [same hyperparameter flags as above]
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram  # noqa: E402

from dataset import prepare_tensors_from_input  # noqa: E402
from analysis.clustering import run_leiden_clustering  # noqa: E402
from learn_mapping import train_spot_to_state, plot_loss_curves  # noqa: E402

# Re-exported so data_preparation.compute_pair_memory (which does
# `from main import estimate_gpu_memory_gb`) keeps working.
from utils import estimate_gpu_memory_gb  # noqa: E402,F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference aggregation and shared-gene helpers
# ---------------------------------------------------------------------------


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


def _median_cosine(Z: torch.Tensor, Z_prime: torch.Tensor, dim: int) -> float:
    """Median cosine similarity along the given dimension (per-spot dim=1, per-gene dim=0)."""
    eps = 1e-8
    dot = torch.sum(Z * Z_prime, dim=dim)
    cos = dot / (torch.norm(Z, p=2, dim=dim) * torch.norm(Z_prime, p=2, dim=dim) + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    return float(torch.median(cos).item())


def _row_one_hot(P: torch.Tensor) -> torch.Tensor:
    """One-hot each row of P at its argmax (deterministic hard assignment)."""
    hard = torch.zeros_like(P)
    hard[torch.arange(P.shape[0]), torch.argmax(P, dim=1)] = 1.0
    return hard


def assemble_state_profiles(
    g_hard: torch.Tensor,
    expr_sums: torch.Tensor,
    sizes: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Size-weighted per-state gene expression profiles from the hard merge g_hard
    (L x K) and the fixed Leiden-cluster expression sums (L x G) / sizes (L,):

        M[k] = (sum_l g[l,k] * expr_sums[l]) / (sum_l g[l,k] * sizes[l])

    Torch mirror of analysis.mapping_metrics.assemble_state_centroids (which does
    the same on numpy for the disk-based analysis).
    """
    weighted_sum = g_hard.t() @ expr_sums  # (K x G)
    state_sizes = g_hard.t() @ sizes  # (K,)
    return weighted_sum / (state_sizes.unsqueeze(1) + eps)


# ---------------------------------------------------------------------------
# Agglomeration tree: cut / merge-cost / one-hot helpers
# ---------------------------------------------------------------------------


def labels_at_k(linkage_z: np.ndarray, k: int, n_leiden: int) -> np.ndarray:
    """Cut the agglomeration tree at k states -> subcluster label array (0..k-1)."""
    if k >= n_leiden:
        return np.arange(n_leiden, dtype=int)
    raw = fcluster(linkage_z, t=k, criterion="maxclust")  # 1..k
    # remap to contiguous 0..k-1
    _, remapped = np.unique(raw, return_inverse=True)
    return remapped.astype(int)


def merge_height_for_k(linkage_z: np.ndarray, k: int, n_leiden: int) -> float:
    """Linkage distance of the merge that produced k clusters (0.0 at k = L)."""
    if k >= n_leiden:
        return 0.0
    # merge row r reduces cluster count to (n_leiden - r - 1); solve for k
    return float(linkage_z[n_leiden - k - 1, 2])


def one_hot_merge(labels_k: np.ndarray, k: int, device: torch.device) -> torch.Tensor:
    """Build the hard subcluster->state matrix G_hard (L x k) from a label array."""
    L = labels_k.shape[0]
    g = torch.zeros(L, k, device=device)
    g[torch.arange(L), torch.as_tensor(labels_k, device=device)] = 1.0
    return g


# ---------------------------------------------------------------------------
# Greedy spot->state mapping (zero-parameter baseline)
# ---------------------------------------------------------------------------


def assign_nearest_centroid(
    Z_shared: torch.Tensor, M_shared: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Assign each spot to the state whose centroid is most cosine-similar to it,
    on the shared genes. No learning.

    Returns a one-hot spot->state assignment P (S x K).
    """
    Zn = Z_shared / (Z_shared.norm(dim=1, keepdim=True) + eps)
    Mn = M_shared / (M_shared.norm(dim=1, keepdim=True) + eps)
    sim = Zn @ Mn.t()  # (S x K) cosine similarity
    spot_state = torch.argmax(sim, dim=1)
    P = torch.zeros(Z_shared.shape[0], M_shared.shape[0], device=Z_shared.device)
    P[torch.arange(P.shape[0], device=Z_shared.device), spot_state] = 1.0
    return P


def _assignment_cosine(
    Z_shared: torch.Tensor, M_shared: torch.Tensor, spot_state: torch.Tensor
) -> torch.Tensor:
    """Per-spot cosine similarity between each spot and the profile of the state
    it is (hard-)assigned to. Works for both greedy and learned mappings."""
    eps = 1e-8
    assigned = M_shared[spot_state]  # (S x G_shared)
    dot = torch.sum(Z_shared * assigned, dim=1)
    cos = dot / (
        torch.norm(Z_shared, p=2, dim=1) * torch.norm(assigned, p=2, dim=1) + eps
    )
    return torch.clamp(cos, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Per-K disk outputs (main.py analysis layout)
# ---------------------------------------------------------------------------


def _write_run_outputs(
    run_dir: Path,
    P: torch.Tensor,
    g_hard: torch.Tensor,
    leiden_labels_np: np.ndarray,
    adata_st: anndata.AnnData,
    adata_sc: anndata.AnnData,
    config: dict,
) -> None:
    """
    Write one K's outputs in the exact layout the post-mapping analysis expects,
    so analysis.run_from_output.analyze_run consumes the folder unchanged:

        mapping_prob.h5ad          — P, obs = spots, var = computed states (S x K)
        leiden_merge_prob.h5ad     — the hard tree cut G, obs = Leiden subclusters,
                                     var = computed states (L x K)
        leiden_overclustering.h5ad — per-cell Leiden label (obs["leiden_cluster"]),
                                     no var
        config.yaml                — provenance; MUST carry
                                     reference_leiden_clustering_resolution, which
                                     analyze_run reads back to match the resolution
                                     the merge was built at.
    """
    L = g_hard.shape[0]
    k = g_hard.shape[1]
    state_names = [f"state_{i}" for i in range(k)]
    leiden_names = [f"leiden_{i}" for i in range(L)]

    anndata.AnnData(
        X=P.detach().cpu().numpy().astype(np.float32),
        obs=pd.DataFrame(index=adata_st.obs_names),
        var=pd.DataFrame(index=state_names),
    ).write_h5ad(run_dir / "mapping_prob.h5ad")

    anndata.AnnData(
        X=g_hard.detach().cpu().numpy().astype(np.float32),
        obs=pd.DataFrame(index=leiden_names),
        var=pd.DataFrame(index=state_names),
    ).write_h5ad(run_dir / "leiden_merge_prob.h5ad")

    cell_cluster_names = [leiden_names[i] for i in leiden_labels_np]
    anndata.AnnData(
        X=np.zeros((len(cell_cluster_names), 0), dtype=np.float32),
        obs=pd.DataFrame(
            {"leiden_cluster": cell_cluster_names}, index=adata_sc.obs_names
        ),
    ).write_h5ad(run_dir / "leiden_overclustering.h5ad")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def run(
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
    mapping: str = "greedy",
    leiden_resolution: float = 3.0,
    normalize_and_log: bool = False,
    epochs: int = 400,
    lr: float = 0.02,
    lambda_spot_gini: float = 1.0,
    spot_gini_warmup_frac: float = 0.5,
    k_min: int | None = None,
    k_max: int | None = None,
    k_step: int = 1,
    full_analysis: bool = False,
) -> pd.DataFrame:
    if mapping not in ("greedy", "learned"):
        raise ValueError(f"mapping must be 'greedy' or 'learned', got {mapping!r}")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s | mapping: %s", device, mapping)

    adata_sc = anndata.read_h5ad(sc_path)
    adata_st = anndata.read_h5ad(st_path)

    # Leiden over-clustering (once, on raw counts).
    logger.info("Computing Leiden over-clustering...")
    labels_np, _ = run_leiden_clustering(adata_sc, resolution=leiden_resolution)
    n_leiden = int(labels_np.max()) + 1
    leiden_labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    logger.info("Leiden clusters: %d", n_leiden)

    if normalize_and_log:
        sc.pp.normalize_total(adata_sc)
        sc.pp.normalize_total(adata_st)
        sc.pp.log1p(adata_sc)
        sc.pp.log1p(adata_st)

    X_shared, Z_shared = prepare_tensors_from_input(adata_sc, adata_st, device)

    # Fixed Leiden aggregates (shared genes) and sizes.
    expr_sums_shared, leiden_sizes = leiden_aggregates(
        X_shared, leiden_labels, n_leiden
    )

    # Shared-gene centroids -> agglomeration tree (average linkage, cosine).
    centroids = (expr_sums_shared / (leiden_sizes.unsqueeze(1) + 1e-8)).cpu().numpy()
    # Guard against all-zero centroids (cosine undefined): give them a tiny uniform value.
    zero_rows = np.where(~centroids.any(axis=1))[0]
    if zero_rows.size:
        centroids[zero_rows] = 1e-6
    linkage_z = linkage(centroids, method="average", metric="cosine")

    k_hi = min(n_leiden, k_max) if k_max else n_leiden
    k_lo = max(1, k_min) if k_min else 1
    ks = list(range(k_hi, k_lo - 1, -k_step))
    logger.info(
        "Sweeping K from %d down to %d (step %d): %d levels (%s mapping)",
        k_hi,
        k_lo,
        k_step,
        len(ks),
        mapping,
    )
    if full_analysis:
        logger.warning(
            "full_analysis=True: running analyze_run for %d K levels "
            "(slow — it recomputes PCA/UMAP/Leiden per K; consider --k_step to "
            "subsample).",
            len(ks),
        )

    rows: list[dict] = []

    for k in ks:
        labels_k = labels_at_k(linkage_z, k, n_leiden)
        g_hard = one_hot_merge(labels_k, k, device)  # (L x k)
        m_shared = assemble_state_profiles(g_hard, expr_sums_shared, leiden_sizes)

        # ---- MODULAR spot->state mapping step --------------------------------
        loss_history = None
        if mapping == "greedy":
            P = assign_nearest_centroid(Z_shared, m_shared)
        else:  # learned
            P, loss_history = train_spot_to_state(
                m_shared,
                Z_shared,
                epochs=epochs,
                lr=lr,
                lambda_spot_gini=lambda_spot_gini,
                device=device,
                spot_gini_warmup_frac=spot_gini_warmup_frac,
            )
        # ----------------------------------------------------------------------

        # Reconstruction (soft P). For greedy, P is one-hot so soft == det below.
        z_prime = torch.matmul(P, m_shared)
        med_spot = _median_cosine(Z_shared, z_prime, dim=1)
        med_gene = _median_cosine(Z_shared, z_prime, dim=0)

        # Deterministic reconstruction: argmax P (the single hard mapping).
        P_hard = _row_one_hot(P)
        z_prime_det = torch.matmul(P_hard, m_shared)
        med_spot_det = _median_cosine(Z_shared, z_prime_det, dim=1)
        med_gene_det = _median_cosine(Z_shared, z_prime_det, dim=0)

        # States actually used by spots (> 1% of spots by argmax).
        spot_state = torch.argmax(P, dim=1)
        frac = torch.bincount(spot_state, minlength=k).float() / P.shape[0]
        n_used = int((frac > 0.01).sum().item())
        n_mapped_states = int(torch.unique(spot_state).numel())

        # One-hotness: mean over spots of the largest per-spot probability
        # (1.0 = every spot maps to exactly one state; always 1.0 for greedy).
        mean_confidence = float(P.max(dim=1).values.mean().item())

        # Mean cosine of each spot to its assigned state profile.
        mean_assign_cos = float(
            _assignment_cosine(Z_shared, m_shared, spot_state).mean().item()
        )

        # Cell-side state occupancy (from the hard merge + cluster sizes).
        state_cell_counts = torch.matmul(g_hard.t(), leiden_sizes)  # (K,)
        cell_frac = state_cell_counts / (state_cell_counts.sum() + 1e-8)
        n_cell_states = int((state_cell_counts > 0).sum().item())
        n_cell_states_gt1pct = int((cell_frac > 0.01).sum().item())

        # Modularities are only produced by the full analysis (graph-based).
        mod_all = float("nan")
        mod_shared = float("nan")
        mod_shared_leiden = float("nan")

        # Per-K outputs — written in the analysis layout so analyze_run reads them.
        run_dir = output_folder / f"K_{k:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if loss_history is not None:
            plot_loss_curves(loss_history, run_dir / "loss_curves.png", k)

        run_config = {
            "method": "agglomerative",
            "mapping": mapping,
            "reference_leiden_clustering_resolution": leiden_resolution,
            "normalize_and_log": normalize_and_log,
            "K": k,
        }
        if mapping == "learned":
            run_config.update(
                {
                    "epochs": epochs,
                    "lr": lr,
                    "lambda_spot_gini": lambda_spot_gini,
                    "spot_gini_warmup_frac": spot_gini_warmup_frac,
                }
            )
        _write_run_outputs(
            run_dir=run_dir,
            P=P,
            g_hard=g_hard,
            leiden_labels_np=labels_np,
            adata_st=adata_st,
            adata_sc=adata_sc,
            config=run_config,
        )

        # subcluster -> state assignment for this level (human-readable).
        pd.DataFrame({"leiden_cluster": np.arange(n_leiden), "state": labels_k}).to_csv(
            run_dir / "leiden_to_state.csv", index=False
        )

        # Spot -> state matrix as CSV (rows = spots), values < 0.01 zeroed then
        # rounded — for eyeballing how close to one-hot the assignment is.
        p_csv = P.detach().cpu().numpy()
        p_csv[p_csv < 0.01] = 0.0
        pd.DataFrame(
            p_csv.round(4),
            index=adata_st.obs_names.tolist(),
            columns=[f"state_{i}" for i in range(k)],
        ).to_csv(run_dir / "P.csv")

        # Optional full post-mapping analysis + report — reuses the decoupled
        # analysis verbatim by reading the folder we just wrote from disk.
        if full_analysis:
            try:
                from analysis.run_from_output import analyze_run

                _results, _n_leiden, objectives = analyze_run(sc_path, st_path, run_dir)
                mod_all = float(objectives.get("modularity_all", float("nan")))
                mod_shared = float(objectives.get("modularity_shared", float("nan")))
                mod_shared_leiden = float(
                    objectives.get("modularity_shared_leiden", float("nan"))
                )
            except Exception as exc:  # keep the sweep going if one K fails
                logger.error("Analysis failed for K=%d: %s", k, exc)

        row = {
            "K": k,
            "merge_height": merge_height_for_k(linkage_z, k, n_leiden),
            "n_cell_states": n_cell_states,
            "n_cell_states_gt1pct": n_cell_states_gt1pct,
            "n_mapped_states": n_mapped_states,
            "n_mapped_states_gt1pct": n_used,
            "median_cossim_spot": round(med_spot, 4),
            "median_cossim_gene": round(med_gene, 4),
            "median_cossim_spot_det": round(med_spot_det, 4),
            "median_cossim_gene_det": round(med_gene_det, 4),
            "mean_spot_confidence": round(mean_confidence, 4),
            "mean_assignment_cosine": round(mean_assign_cos, 4),
            "modularity_all": round(mod_all, 4),
            "modularity_shared": round(mod_shared, 4),
            "modularity_shared_leiden": round(mod_shared_leiden, 4),
        }
        rows.append(row)
        logger.info(
            "K=%3d | cos_spot=%.4f cos_gene=%.4f | used=%d | conf=%.3f | "
            "mean_assign_cos=%.4f | merge_h=%.4f",
            k,
            med_spot,
            med_gene,
            n_used,
            mean_confidence,
            mean_assign_cos,
            row["merge_height"],
        )

    summary = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    summary.to_csv(output_folder / "summary.csv", index=False)
    logger.info("Wrote %s", output_folder / "summary.csv")

    _make_decision_plots(summary, linkage_z, output_folder)
    _make_summary_report(summary, output_folder)
    return summary


# ---------------------------------------------------------------------------
# Cross-K plots
# ---------------------------------------------------------------------------


def _make_summary_report(summary: pd.DataFrame, output_folder: Path) -> None:
    """Multi-panel cross-K overview: reconstruction/confidence, state counts,
    modularity, and merge cost. Writes summary_across_k.png."""
    ks = summary["K"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reconstruction + assignment confidence (all in [0, 1]).
    ax = axes[0, 0]
    ax.plot(ks, summary["median_cossim_spot"], "-o", ms=3, label="cos spot (soft)")
    ax.plot(ks, summary["median_cossim_gene"], "-s", ms=3, label="cos gene (soft)")
    ax.plot(
        ks, summary["median_cossim_spot_det"], "--", alpha=0.6, label="cos spot (det)"
    )
    ax.plot(
        ks, summary["median_cossim_gene_det"], "--", alpha=0.6, label="cos gene (det)"
    )
    ax.plot(ks, summary["mean_spot_confidence"], "-^", ms=3, label="spot confidence")
    ax.set_title("Reconstruction & confidence")
    ax.set_xlabel("K")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # State counts.
    ax = axes[0, 1]
    ax.plot(ks, summary["n_cell_states"], "-o", ms=3, label="cell states")
    ax.plot(ks, summary["n_cell_states_gt1pct"], "-s", ms=3, label="cell states >1%")
    ax.plot(ks, summary["n_mapped_states"], "-^", ms=3, label="mapped states")
    ax.plot(
        ks, summary["n_mapped_states_gt1pct"], "-d", ms=3, label="mapped states >1%"
    )
    ax.set_title("State counts")
    ax.set_xlabel("K")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Modularity (only when the full analysis ran).
    ax = axes[1, 0]
    mod_cols = ["modularity_all", "modularity_shared", "modularity_shared_leiden"]
    if summary[mod_cols].notna().any().any():
        ax.plot(ks, summary["modularity_all"], "-o", ms=3, label="all genes")
        ax.plot(ks, summary["modularity_shared"], "-s", ms=3, label="shared genes")
        ax.plot(
            ks,
            summary["modularity_shared_leiden"],
            "--",
            color="gray",
            label="Leiden-shared ceiling",
        )
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "modularity requires --full_analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
    ax.set_title("Modularity of computed cell states")
    ax.set_xlabel("K")
    ax.grid(True, alpha=0.3)

    # Merge cost.
    ax = axes[1, 1]
    ax.plot(ks, summary["merge_height"], "-^", ms=3, color="tab:red")
    ax.set_title("Merge cost (K+1 -> K)")
    ax.set_xlabel("K")
    ax.set_ylabel("cosine distance")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Agglomerative-K summary")
    fig.tight_layout()
    fig.savefig(output_folder / "summary_across_k.png", dpi=150)
    plt.close(fig)


def _make_decision_plots(
    summary: pd.DataFrame, linkage_z: np.ndarray, output_folder: Path
) -> None:
    """Reconstruction-vs-K, merge-cost-vs-K, a combined tradeoff plot, and the dendrogram."""
    ks = summary["K"].to_numpy()

    # Reconstruction vs K.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        ks, summary["median_cossim_spot"], "-o", ms=3, label="median cos (per spot)"
    )
    ax.plot(
        ks, summary["median_cossim_gene"], "-s", ms=3, label="median cos (per gene)"
    )
    ax.plot(
        ks,
        summary["median_cossim_spot_det"],
        "--",
        alpha=0.6,
        label="median cos spot (deterministic)",
    )
    ax.plot(
        ks,
        summary["median_cossim_gene_det"],
        "--",
        alpha=0.6,
        label="median cos gene (deterministic)",
    )
    ax.plot(
        ks,
        summary["mean_spot_confidence"],
        "-^",
        ms=3,
        color="tab:green",
        label="mean spot confidence (one-hotness)",
    )
    ax.set_xlabel("K (number of states)")
    ax.set_ylabel("value in [0, 1]")
    ax.set_ylim(0, 1.02)
    ax.set_title("Reconstruction & assignment confidence vs K")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "recon_vs_k.png", dpi=150)
    plt.close(fig)

    # Combined tradeoff: reconstruction (left) + merge cost (right).
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(
        ks,
        summary["median_cossim_spot"],
        "-o",
        ms=3,
        color="tab:blue",
        label="median cos (per spot)",
    )
    ax1.set_xlabel("K (number of states)")
    ax1.set_ylabel("median cosine (per spot)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(
        ks,
        summary["merge_height"],
        "-^",
        ms=3,
        color="tab:red",
        label="merge cost (K+1 -> K)",
    )
    ax2.set_ylabel("merge cost (cosine dist)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Fit vs parsimony: pick K where fit plateaus but merges stay cheap")
    fig.tight_layout()
    fig.savefig(output_folder / "tradeoff_vs_k.png", dpi=150)
    plt.close(fig)

    # Dendrogram.
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_z, ax=ax, no_labels=True, color_threshold=0.0)
    ax.set_title(
        "Agglomeration of Leiden subclusters (shared-gene cosine, average linkage)"
    )
    ax.set_ylabel("merge distance")
    fig.tight_layout()
    fig.savefig(output_folder / "dendrogram.png", dpi=150)
    plt.close(fig)

    logger.info("Wrote decision plots to %s", output_folder)


# ---------------------------------------------------------------------------
# Single-pair / batch drivers
# ---------------------------------------------------------------------------


def _run_one_pair(
    sc_path: Path, st_path: Path, output_folder: Path, args
) -> pd.DataFrame:
    """Write config.yaml + run the K-sweep for a single sc/st pair."""
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "method": "agglomerative",
                "mapping": args.mapping,
                "leiden_resolution": args.leiden_resolution,
                "normalize_and_log": args.normalize_and_log,
                "epochs": args.epochs,
                "lr": args.lr,
                "lambda_spot_gini": args.lambda_spot_gini,
                "spot_gini_warmup_frac": args.spot_gini_warmup_frac,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "k_step": args.k_step,
                "full_analysis": args.full_analysis,
            },
            f,
            sort_keys=False,
        )

    return run(
        sc_path=sc_path,
        st_path=st_path,
        output_folder=output_folder,
        mapping=args.mapping,
        leiden_resolution=args.leiden_resolution,
        normalize_and_log=args.normalize_and_log,
        epochs=args.epochs,
        lr=args.lr,
        lambda_spot_gini=args.lambda_spot_gini,
        spot_gini_warmup_frac=args.spot_gini_warmup_frac,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
        full_analysis=args.full_analysis,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agglomerative-K sweep — modular greedy / learned spot-to-state mapping"
    )
    # Single-pair mode
    parser.add_argument(
        "--scdata", type=Path, default=None, help="Single-pair mode: sc .h5ad path"
    )
    parser.add_argument(
        "--stdata", type=Path, default=None, help="Single-pair mode: ST .h5ad path"
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=None,
        help="Single-pair mode: output folder",
    )
    # Batch mode
    parser.add_argument(
        "--pairs_csv", type=Path, default=None, help="Batch mode: path to pairs.csv"
    )
    parser.add_argument(
        "--sc_dir",
        type=Path,
        default=None,
        help="Batch mode: folder containing scRNA .h5ad files",
    )
    parser.add_argument(
        "--st_dir",
        type=Path,
        default=None,
        help="Batch mode: folder containing ST .h5ad files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Batch mode: root folder for all pair outputs",
    )
    # Method knobs
    parser.add_argument(
        "--mapping",
        choices=["greedy", "learned"],
        default="greedy",
        help="Spot-to-state mapping: 'greedy' (zero-parameter nearest-centroid, "
        "default) or 'learned' (gradient-descent soft P). The learned-mode flags "
        "below are ignored when --mapping greedy.",
    )
    parser.add_argument("--leiden_resolution", type=float, default=3.0)
    parser.add_argument("--normalize_and_log", action="store_true", default=False)
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--k_step", type=int, default=1)
    parser.add_argument(
        "--full_analysis",
        action="store_true",
        default=False,
        help="Run analysis.run_from_output.analyze_run for every K (slow)",
    )
    # Learned-mode only
    parser.add_argument(
        "--epochs", type=int, default=400, help="[learned] per-K deconvolution epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.02, help="[learned] learning rate"
    )
    parser.add_argument(
        "--lambda_spot_gini",
        type=float,
        default=1.0,
        help="[learned] quadratic (Gini/Tsallis-2) spot sharpener on P — the strong "
        "one-hot lever. Set 0 to disable.",
    )
    parser.add_argument(
        "--spot_gini_warmup_frac",
        type=float,
        default=0.5,
        help="[learned] fraction of epochs to train with spot_gini OFF before "
        "ramping it in linearly (e.g. 0.5 = pure reconstruction for the first half, "
        "then ramp to full by the last epoch). 0 = constant weight from the start.",
    )
    parser.add_argument("--logging", choices=["normal", "verbose"], default="normal")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.logging == "verbose" else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    batch_mode = args.pairs_csv is not None
    if batch_mode:
        if args.sc_dir is None or args.st_dir is None or args.output_dir is None:
            parser.error("--pairs_csv requires --sc_dir, --st_dir, and --output_dir")

        with open(args.pairs_csv, newline="") as fh:
            all_pairs = list(csv.DictReader(fh))
        logger.info("Loaded %d pairs from %s", len(all_pairs), args.pairs_csv)

        errors: list[str] = []
        for pair in all_pairs:
            pair_id = int(pair["PairID"])
            sc_name = pair["scName"]
            st_name = pair["stName"]
            sc_path = args.sc_dir / f"{sc_name}.h5ad"
            st_path = args.st_dir / f"{st_name}.h5ad"

            missing = [p for p in (sc_path, st_path) if not p.exists()]
            if missing:
                msg = f"[Pair {pair_id:>3}] Missing files: {[str(p) for p in missing]}"
                logger.error(msg)
                errors.append(msg)
                continue

            pair_output = args.output_dir / f"{pair_id:03d}_{sc_name}__{st_name}"
            logger.info(
                "[Pair %3d] %s x %s -> %s", pair_id, sc_name, st_name, pair_output
            )
            try:
                _run_one_pair(sc_path, st_path, pair_output, args)
            except Exception as exc:
                msg = f"[Pair {pair_id:>3}] FAILED: {exc}"
                logger.error(msg)
                errors.append(msg)

        if errors:
            logger.warning("%d pair(s) failed:", len(errors))
            for e in errors:
                logger.warning("  %s", e)
        else:
            logger.info("All pairs completed successfully.")
    else:
        if args.scdata is None or args.stdata is None or args.output_folder is None:
            parser.error(
                "Provide either --scdata/--stdata/--output_folder (single pair) "
                "or --pairs_csv/--sc_dir/--st_dir/--output_dir (batch)"
            )
        _run_one_pair(args.scdata, args.stdata, args.output_folder, args)
