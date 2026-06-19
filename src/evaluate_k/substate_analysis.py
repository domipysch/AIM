"""
Per-state sub-cluster analysis.

For each computed state, finds the Leiden sub-clusters that map to it and
computes four diagnostic metrics derived from the validation notebook
`validate_7_3/computed_vs_leiden_states.ipynb`.

Public API
----------
compute_substate_metrics(adata_sc, adata_st, computed_states, spot_states,
                         leiden_labels, shared_genes, ...)
    -> dict with keys "per_state", "weighted_perm_p"
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _dense(mat) -> np.ndarray:
    return mat.toarray().astype(float) if issparse(mat) else np.array(mat, dtype=float)


def _mean_pairwise_cossim(vecs: np.ndarray) -> float:
    """Mean cosine similarity over all unique pairs of rows."""
    n = len(vecs)
    if n < 2:
        return 1.0
    sims = [
        float(cosine_similarity(vecs[i].reshape(1, -1), vecs[j].reshape(1, -1))[0, 0])
        for i, j in combinations(range(n), 2)
    ]
    return float(np.mean(sims))


# ─── Main function ─────────────────────────────────────────────────────────────


def compute_substate_metrics(
    adata_sc: AnnData,
    adata_st: AnnData,
    computed_states: np.ndarray,
    spot_states: np.ndarray,
    leiden_labels: np.ndarray,
    shared_genes: list[str],
    n_top_genes: int = 30,
    n_perm: int = 50,
    seed: int = 42,
) -> dict:
    """
    Compute per-state sub-cluster diagnostics for every computed state.

    Parameters
    ----------
    adata_sc          : Single-cell AnnData (cells × genes), raw counts.
    adata_st          : Spatial AnnData (spots × shared_genes).
    computed_states   : Hard cell-state labels, shape (n_cells,).
    spot_states       : Hard spot-state labels, shape (n_spots,).
    leiden_labels     : Leiden cluster labels on sc data, shape (n_cells,).
    shared_genes      : Genes present in both sc and st data (ordered).
    n_top_genes       : Number of top highly-variable genes (by variance across all ST spots) used for cosine similarity.
    n_perm            : Number of permutations for null distributions.
    seed              : RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        "per_state"        dict[cs_id → metrics dict], each entry containing:
                               n_cells, n_spots, n_leiden_sub,
                               cossim_centroid, perm_p_value
        "weighted_perm_p"  float — spot-weighted mean perm_p across states with
                               n_leiden_sub > 1; nan if no such states exist.
    """
    rng = np.random.default_rng(seed)

    # ── Dense matrices ────────────────────────────────────────────────────────
    X_sc = _dense(adata_sc.X)  # (n_cells, G_sc)
    X_st_full = _dense(adata_st.X)  # (n_spots, G_st)

    sc_gene_names = list(adata_sc.var_names)
    st_gene_names = list(adata_st.var_names)

    shared_st_idx = [st_gene_names.index(g) for g in shared_genes]

    X_st_shared = X_st_full[:, shared_st_idx]  # (n_spots, n_shared)

    # Gene selection is now per-state (see inside the loop below).

    # ── Map Leiden clusters to computed states ────────────────────────────────
    unique_cs = sorted(np.unique(computed_states).tolist())
    all_leiden = np.unique(leiden_labels)

    # For each Leiden cluster: which computed state owns the most cells?
    ls_per_cs: dict[int, list[int]] = {cs: [] for cs in unique_cs}
    for ls in all_leiden:
        mask = leiden_labels == ls
        votes = computed_states[mask]
        if len(votes) == 0:
            continue
        dominant_cs = int(np.bincount(votes, minlength=max(unique_cs) + 1).argmax())
        ls_per_cs[dominant_cs].append(int(ls))

    # ── Per-state loop ────────────────────────────────────────────────────────
    per_state: dict[int, dict] = {}

    for cs in unique_cs:
        ls_targets = sorted(ls_per_cs[cs])
        other_leiden = [int(l) for l in all_leiden if l not in ls_targets]

        cs_cell_mask = computed_states == cs
        cs_spot_mask = spot_states == cs
        n_cells = int(cs_cell_mask.sum())
        n_spots = int(cs_spot_mask.sum())

        base = {
            "n_cells": n_cells,
            "n_spots": n_spots,
            "n_leiden_sub": len(ls_targets),
        }

        if len(ls_targets) == 0 or n_spots == 0:
            logger.debug(
                "CS%d: skipping sub-cluster metrics (n_leiden_sub=%d, n_spots=%d)",
                cs,
                len(ls_targets),
                n_spots,
            )
            per_state[cs] = {
                **base,
                "cossim_centroid": float("nan"),
                "perm_p_value": float("nan"),
            }
            continue

        logger.debug(
            "CS%d: %d Leiden sub-clusters, %d spots", cs, len(ls_targets), n_spots
        )

        # ── Per-state gene selection: top-N by fold-change vs all other spots ─
        mean_state = X_st_shared[cs_spot_mask].mean(axis=0)  # (n_shared,)
        mean_rest = X_st_shared[~cs_spot_mask].mean(axis=0)  # (n_shared,)
        fold_change = mean_state / (mean_rest + 1e-6)
        top_n_idx = np.argsort(fold_change)[::-1][:n_top_genes]
        top_n_sc_idx = [sc_gene_names.index(shared_genes[i]) for i in top_n_idx]
        logger.debug(
            "CS%d: top-%d genes by fold-change: %s…",
            cs,
            n_top_genes,
            [shared_genes[i] for i in top_n_idx[:5]],
        )

        # Leiden centroids in this state's gene subspace
        leiden_centroids_state: dict[int, np.ndarray] = {
            int(ls): X_sc[leiden_labels == ls][:, top_n_sc_idx].mean(axis=0)
            for ls in all_leiden
        }

        # ── cossim_centroid + perm_p_value ────────────────────────────────────
        ls_vecs_topn = np.array(
            [leiden_centroids_state[ls] for ls in ls_targets]
        )  # (n_ls, n_top_genes)
        cossim_centroid = _mean_pairwise_cossim(ls_vecs_topn)

        if len(ls_targets) == 1:
            # Single sub-cluster: perfectly coherent by definition → perm_p = 0.0
            perm_p_value = 0.0
        else:
            n_draw = len(ls_targets)
            replace = n_draw > len(other_leiden)
            perm_draws = [
                rng.choice(other_leiden, size=n_draw, replace=replace).tolist()
                for _ in range(n_perm)
            ]
            perm_sims_a = []
            for draw in perm_draws:
                vecs = np.array([leiden_centroids_state[int(c)] for c in draw])
                perm_sims_a.append(_mean_pairwise_cossim(vecs))
            perm_p_value = float(np.mean(np.array(perm_sims_a) >= cossim_centroid))

        per_state[cs] = {
            **base,
            "cossim_centroid": float(cossim_centroid),
            "perm_p_value": float(perm_p_value),
        }

    # ── Weighted average perm_p (spot-weighted, mapped states only) ───────────
    total_spots = 0
    weighted_sum = 0.0
    for m in per_state.values():
        n_spots = m["n_spots"]
        p = m["perm_p_value"]
        if n_spots > 0 and m["n_leiden_sub"] > 1 and p == p:  # exclude trivial/NaN
            weighted_sum += p * n_spots
            total_spots += n_spots
    weighted_perm_p = weighted_sum / total_spots if total_spots > 0 else float("nan")

    logger.info(
        "Substate analysis done: %d states  |  weighted_perm_p=%.4f",
        len(per_state),
        weighted_perm_p,
    )

    return {
        "per_state": per_state,
        "weighted_perm_p": weighted_perm_p,
    }
