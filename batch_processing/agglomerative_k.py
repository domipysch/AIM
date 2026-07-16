"""
Agglomerative-K sweep for AIM (leiden-merge-mapping setup).

An interpretable alternative to jointly learning the merge matrix G. Instead of
letting a loss decide how Leiden over-clusters combine, we:

    1. Leiden over-cluster the reference -> L subclusters (+ centroids, sizes).
    2. Build the full agglomeration tree ONCE with average-linkage on the
       shared-gene cosine distance between centroids (scipy.linkage). This *is*
       "merge the two closest, recompute, repeat" — scipy does the recomputation.
    3. For every K from L down to 1: cut the tree at K states, assemble the
       (size-weighted) merged state profiles, and train ONLY a spot->state map P
       (a deconvolution against the fixed profiles). The hard tree cut plays the
       role of G; only P is learned.
    4. Emit a summary.csv and decision plots (reconstruction vs K, merge cost vs
       K, dendrogram) so the user can pick the K they want.

The merge order here is decided by shared-gene expression distance alone; the
spatial reconstruction only *scores* each K (it does not choose the merges).

This mirrors the new decoupled architecture: each K's folder is written in the
exact same layout main.py produces (mapping_prob.h5ad = P, leiden_merge_prob.h5ad
= the hard cut G, leiden_overclustering.h5ad, config.yaml). The full post-mapping
analysis is therefore just analysis.run_from_output.analyze_run on each K folder,
reused verbatim (with --full_analysis).

Run:
    PYTHONPATH=src python -m batch_processing.agglomerative_k \
        --scdata <sc.h5ad> --stdata <st.h5ad> --output_folder <out> \
        [--leiden_resolution 3.0] [--epochs 400] [--lr 0.02] \
        [--lambda_spot_entropy 0.2] [--k_min 1] [--k_max <L>] [--k_step 1] \
        [--normalize_and_log] [--full_analysis]
"""

import argparse
import logging
import math
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

import main as aim_main  # noqa: E402  (leiden_aggregates)
from dataset import prepare_tensors_from_input  # noqa: E402
from analysis.clustering import run_leiden_clustering  # noqa: E402

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


def train_spot_to_state(
    M_shared: torch.Tensor,
    Z_shared: torch.Tensor,
    epochs: int,
    lr: float,
    lambda_spot_entropy: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """
    Learn ONLY the spot->state matrix P (S x K) that reconstructs the ST data
    from fixed state profiles M_shared (K x G_shared): minimize spot + gene
    cosine distance, with an optional per-spot entropy penalty.

    Returns:
        P:       spot->state soft assignment (S x K), rows sum to 1.
        history: per-epoch unweighted values of each loss component
                 (rec_spot, rec_gene, spot_entropy).
    """
    n_spots = Z_shared.shape[0]
    n_states = M_shared.shape[0]
    logits = torch.nn.Parameter(torch.randn(n_spots, n_states, device=device))

    optimizer = torch.optim.Adam([logits], lr=lr)
    ln_k = math.log(n_states) if n_states > 1 else 1.0

    history: dict[str, list[float]] = {
        "rec_spot": [],
        "rec_gene": [],
        "spot_entropy": [],
    }
    for _ in range(epochs):
        optimizer.zero_grad()
        P = torch.softmax(logits, dim=1)
        Z_prime = torch.matmul(P, M_shared)  # (S x G_shared)
        l_rec_spot = _cosine_distance_loss(Z_shared, Z_prime, dim=1)
        l_rec_gene = _cosine_distance_loss(Z_shared, Z_prime, dim=0)
        # Always compute the entropy (for the curve); only penalize it if enabled.
        entropy = -torch.sum(P * torch.log(P + 1e-8), dim=1).mean() / ln_k
        loss = l_rec_spot + l_rec_gene
        if lambda_spot_entropy > 0.0 and n_states > 1:
            loss = loss + lambda_spot_entropy * entropy
        loss.backward()
        optimizer.step()
        history["rec_spot"].append(float(l_rec_spot.detach()))
        history["rec_gene"].append(float(l_rec_gene.detach()))
        history["spot_entropy"].append(float(entropy.detach()))

    return torch.softmax(logits.detach(), dim=1), history


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


def _write_run_outputs(
    run_dir: Path,
    P: torch.Tensor,
    g_hard: torch.Tensor,
    leiden_labels_np: np.ndarray,
    adata_st: anndata.AnnData,
    adata_sc: anndata.AnnData,
    leiden_resolution: float,
    normalize_and_log: bool,
    k: int,
    epochs: int,
    lr: float,
    lambda_spot_entropy: float,
) -> None:
    """
    Write one K's outputs in the exact layout main.py produces, so that
    analysis.run_from_output.analyze_run can consume the folder unchanged:

        mapping_prob.h5ad       — P, obs = spots, var = computed states (S x K)
        leiden_merge_prob.h5ad  — the hard tree cut G, obs = Leiden subclusters,
                                  var = computed states (L x K)
        leiden_overclustering.h5ad — per-cell Leiden label (obs["leiden_cluster"]
                                  = "leiden_<i>"), no var
        config.yaml             — provenance; MUST carry
                                  reference_leiden_clustering_resolution, which
                                  analyze_run reads back to match the resolution
                                  the merge was built at.
    """
    L = g_hard.shape[0]
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
        yaml.safe_dump(
            {
                "method": "agglomerative_k",
                "reference_leiden_clustering_resolution": leiden_resolution,
                "normalize_and_log": normalize_and_log,
                "K": k,
                "epochs": epochs,
                "lr": lr,
                "lambda_spot_entropy": lambda_spot_entropy,
            },
            f,
            sort_keys=False,
        )


def run(
    sc_path: Path,
    st_path: Path,
    output_folder: Path,
    leiden_resolution: float = 3.0,
    normalize_and_log: bool = False,
    epochs: int = 400,
    lr: float = 0.02,
    lambda_spot_entropy: float = 0.2,
    k_min: int | None = None,
    k_max: int | None = None,
    k_step: int = 1,
    full_analysis: bool = False,
) -> pd.DataFrame:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    adata_sc = anndata.read_h5ad(sc_path)
    adata_st = anndata.read_h5ad(st_path)

    # Leiden over-clustering (once, on raw counts — mirrors main.py's order).
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
    expr_sums_shared, leiden_sizes = aim_main.leiden_aggregates(
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
        "Sweeping K from %d down to %d (step %d): %d levels",
        k_hi,
        k_lo,
        k_step,
        len(ks),
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

        # Fixed state profiles on shared genes -> deconvolve spots against them.
        m_shared = assemble_state_profiles(g_hard, expr_sums_shared, leiden_sizes)
        P, loss_history = train_spot_to_state(
            m_shared, Z_shared, epochs, lr, lambda_spot_entropy, device
        )

        # Reconstruction quality on shared genes (probabilistic / soft P).
        z_prime = torch.matmul(P, m_shared)
        med_spot = _median_cosine(Z_shared, z_prime, dim=1)
        med_gene = _median_cosine(Z_shared, z_prime, dim=0)

        # Deterministic reconstruction: argmax P (G is already hard) — what the
        # single hard mapping actually delivers.
        z_prime_det = torch.matmul(_row_one_hot(P), m_shared)
        med_spot_det = _median_cosine(Z_shared, z_prime_det, dim=1)
        med_gene_det = _median_cosine(Z_shared, z_prime_det, dim=0)

        # States actually used by spots (> 1% of spots by argmax).
        spot_state = torch.argmax(P, dim=1)
        frac = torch.bincount(spot_state, minlength=k).float() / P.shape[0]
        n_used = int((frac > 0.01).sum().item())
        n_mapped_states = int(torch.unique(spot_state).numel())

        # One-hotness: mean over spots of the largest per-spot probability
        # (1.0 = every spot maps to exactly one state).
        mean_confidence = float(P.max(dim=1).values.mean().item())

        # Cell-side state occupancy (from the hard merge + cluster sizes).
        state_cell_counts = torch.matmul(g_hard.t(), leiden_sizes)  # (K,)
        cell_frac = state_cell_counts / (state_cell_counts.sum() + 1e-8)
        n_cell_states = int((state_cell_counts > 0).sum().item())
        n_cell_states_gt1pct = int((cell_frac > 0.01).sum().item())

        # Modularities are only produced by the full analysis (graph-based).
        mod_all = float("nan")
        mod_shared = float("nan")
        mod_shared_leiden = float("nan")

        # Per-K outputs — written in main.py's layout so analyze_run can read them.
        run_dir = output_folder / f"K_{k:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _plot_loss_curves(loss_history, run_dir / "loss_curves.png", k)
        _write_run_outputs(
            run_dir=run_dir,
            P=P,
            g_hard=g_hard,
            leiden_labels_np=labels_np,
            adata_st=adata_st,
            adata_sc=adata_sc,
            leiden_resolution=leiden_resolution,
            normalize_and_log=normalize_and_log,
            k=k,
            epochs=epochs,
            lr=lr,
            lambda_spot_entropy=lambda_spot_entropy,
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
            "modularity_all": round(mod_all, 4),
            "modularity_shared": round(mod_shared, 4),
            "modularity_shared_leiden": round(mod_shared_leiden, 4),
        }
        rows.append(row)
        logger.info(
            "K=%3d | cos_spot=%.4f cos_gene=%.4f | used=%d | conf=%.3f | merge_h=%.4f",
            k,
            med_spot,
            med_gene,
            n_used,
            mean_confidence,
            row["merge_height"],
        )

    summary = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    summary.to_csv(output_folder / "summary.csv", index=False)
    logger.info("Wrote %s", output_folder / "summary.csv")

    _make_decision_plots(summary, linkage_z, output_folder)
    _make_summary_report(summary, output_folder)
    return summary


def _plot_loss_curves(history: dict[str, list[float]], out_path: Path, k: int) -> None:
    """Plot the per-epoch deconvolution loss components for one K."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(len(history["rec_spot"]))
    ax.plot(epochs, history["rec_spot"], label="rec_spot (1 - cos, per spot)")
    ax.plot(epochs, history["rec_gene"], label="rec_gene (1 - cos, per gene)")
    ax.plot(epochs, history["spot_entropy"], label="spot_entropy (normalised)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss component (unweighted)")
    ax.set_title(f"Deconvolution loss components (K = {k})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agglomerative-K sweep for AIM")
    parser.add_argument("--scdata", type=Path, required=True)
    parser.add_argument("--stdata", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--leiden_resolution", type=float, default=3.0)
    parser.add_argument("--normalize_and_log", action="store_true", default=False)
    parser.add_argument(
        "--epochs", type=int, default=400, help="Per-K deconvolution epochs"
    )
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lambda_spot_entropy", type=float, default=0.2)
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--k_step", type=int, default=1)
    parser.add_argument(
        "--full_analysis",
        action="store_true",
        default=False,
        help="Run analysis.run_from_output.analyze_run for every K (slow)",
    )
    parser.add_argument("--logging", choices=["normal", "verbose"], default="normal")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.logging == "verbose" else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args.output_folder.mkdir(parents=True, exist_ok=True)
    with open(args.output_folder / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "leiden_resolution": args.leiden_resolution,
                "normalize_and_log": args.normalize_and_log,
                "epochs": args.epochs,
                "lr": args.lr,
                "lambda_spot_entropy": args.lambda_spot_entropy,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "k_step": args.k_step,
                "full_analysis": args.full_analysis,
            },
            f,
            sort_keys=False,
        )

    run(
        sc_path=args.scdata,
        st_path=args.stdata,
        output_folder=args.output_folder,
        leiden_resolution=args.leiden_resolution,
        normalize_and_log=args.normalize_and_log,
        epochs=args.epochs,
        lr=args.lr,
        lambda_spot_entropy=args.lambda_spot_entropy,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
        full_analysis=args.full_analysis,
    )
