"""
Per-state sub-cluster analysis.

For each computed state, finds the Leiden sub-clusters that map to it and
computes four diagnostic metrics derived from the validation notebook
`validate_7_3/computed_vs_leiden_states.ipynb`.

Public API
----------
compute_substate_metrics(adata_sc, adata_st, computed_states, spot_states,
                         leiden_labels, shared_genes, ...)
    -> dict with keys "per_state", "margin_computed_mean", "margin_leiden_mean"
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


def _margin(sim_mat: np.ndarray) -> np.ndarray:
    """Per-spot margin: cosim(winner) − cosim(runner-up). Shape (n_spots,)."""
    top2 = np.partition(sim_mat, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


# ─── Main function ─────────────────────────────────────────────────────────────


def compute_substate_metrics(
    adata_sc: AnnData,
    adata_st: AnnData,
    computed_states: np.ndarray,
    spot_states: np.ndarray,
    leiden_labels: np.ndarray,
    shared_genes: list[str],
    n_top_genes: int = 8,
    n_perm: int = 100,
    n_top_discriminable: int = 100,
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
    n_top_genes       : Number of top genes to select from spot centroid.
    n_perm            : Number of permutations for null distributions.
    n_top_discriminable : Top-N discriminable genes to check membership in.
    seed              : RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        "per_state"            dict[cs_id -> metrics dict]
        "margin_computed_mean" float  mean spot margin under computed clustering
        "margin_leiden_mean"   float  mean spot margin under Leiden clustering
    """
    rng = np.random.default_rng(seed)

    # ── Dense matrices ────────────────────────────────────────────────────────
    X_sc = _dense(adata_sc.X)  # (n_cells, G_sc)
    X_st_full = _dense(adata_st.X)  # (n_spots, G_st)

    sc_gene_names = list(adata_sc.var_names)
    st_gene_names = list(adata_st.var_names)

    shared_sc_idx = [sc_gene_names.index(g) for g in shared_genes]
    shared_st_idx = [st_gene_names.index(g) for g in shared_genes]

    X_st_shared = X_st_full[:, shared_st_idx]  # (n_spots, n_shared)

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
                "marker_in_top100_count": float("nan"),
                "indisting_actual_mean": float("nan"),
                "indisting_null_mean": float("nan"),
            }
            continue

        logger.debug(
            "CS%d: %d Leiden sub-clusters, %d spots", cs, len(ls_targets), n_spots
        )

        ls_masks = {ls: leiden_labels == ls for ls in ls_targets}

        # ── Spot centroid → top-N marker genes ────────────────────────────────
        cs_spots_shared = X_st_shared[cs_spot_mask]  # (n_spots, n_shared)
        centroid_spots = cs_spots_shared.mean(axis=0)  # (n_shared,)

        top_n_st_idx = np.argsort(centroid_spots)[::-1][:n_top_genes]
        top_n_genes = [shared_genes[i] for i in top_n_st_idx]
        top_n_sc_idx = [sc_gene_names.index(g) for g in top_n_genes]

        # ── LS centroids (full gene space + shared gene space) ────────────────
        ls_centroids_full: dict[int, np.ndarray] = {
            ls: X_sc[mask].mean(axis=0) for ls, mask in ls_masks.items()
        }
        ls_centroids_shared_arr = np.array(
            [ls_centroids_full[ls][shared_sc_idx] for ls in ls_targets]
        )  # (n_ls, n_shared)

        # ── Metric A: cossim_centroid + perm_p_value ─────────────────────────
        ls_vecs_topn = np.array(
            [ls_centroids_full[ls][top_n_sc_idx] for ls in ls_targets]
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
                vecs = np.array(
                    [
                        X_sc[leiden_labels == c][:, top_n_sc_idx].mean(axis=0)
                        for c in draw
                    ]
                )
                perm_sims_a.append(_mean_pairwise_cossim(vecs))
            perm_p_value = float(np.mean(np.array(perm_sims_a) >= cossim_centroid))

        # ── Metric B: fraction of top-N genes in top-discriminable genes ──────
        # Requires at least 2 sub-clusters to form pairwise log2FC comparisons.
        if len(ls_targets) >= 2:
            ls_pairs = list(combinations(ls_targets, 2))
            pair_log2fc = np.zeros((len(ls_pairs), len(sc_gene_names)))
            for k, (ls_i, ls_j) in enumerate(ls_pairs):
                X_i = X_sc[ls_masks[ls_i]]
                X_j = X_sc[ls_masks[ls_j]]
                pair_log2fc[k] = np.abs(
                    np.log2(X_i.mean(axis=0) + 1) - np.log2(X_j.mean(axis=0) + 1)
                )
            gene_discrim = pair_log2fc.mean(axis=0)
            top_discrim_set = set(
                np.argsort(gene_discrim)[::-1][:n_top_discriminable].tolist()
            )
            n_in_top = len(set(top_n_sc_idx) & top_discrim_set)
            marker_in_top100_count: int | float = int(n_in_top)
        else:
            marker_in_top100_count = float("nan")

        # ── Metric C: per-spot indistinguishability ───────────────────────────
        # Requires at least 2 sub-clusters; variance over a single centroid is
        # always 0 and the null is identical, so the comparison carries no information.
        if len(ls_targets) >= 2:
            sim_matrix = cosine_similarity(
                cs_spots_shared, ls_centroids_shared_arr
            )  # (n_spots, n_ls)
            spot_sim_var = sim_matrix.var(axis=1)
            indisting_actual_mean = float(spot_sim_var.mean())

            perm_null_means = []
            for draw in perm_draws:
                rand_cents = np.array(
                    [
                        X_sc[leiden_labels == c][:, shared_sc_idx].mean(axis=0)
                        for c in draw
                    ]
                )
                sim_perm = cosine_similarity(cs_spots_shared, rand_cents)
                perm_null_means.append(float(sim_perm.var(axis=1).mean()))
            indisting_null_mean = float(np.mean(perm_null_means))
        else:
            indisting_actual_mean = float("nan")
            indisting_null_mean = float("nan")

        per_state[cs] = {
            **base,
            "cossim_centroid": float(cossim_centroid),
            "perm_p_value": float(perm_p_value),
            "marker_in_top100_count": marker_in_top100_count,
            "indisting_actual_mean": indisting_actual_mean,
            "indisting_null_mean": indisting_null_mean,
        }

    # ── Global: spot-level mapping clarity ────────────────────────────────────
    unique_cs_arr = np.array(unique_cs)
    cs_cents = np.array(
        [
            X_sc[computed_states == cs][:, shared_sc_idx].mean(axis=0)
            for cs in unique_cs_arr
        ]
    )  # (K, n_shared)

    unique_ld = np.unique(leiden_labels)
    ld_cents = np.array(
        [X_sc[leiden_labels == ld][:, shared_sc_idx].mean(axis=0) for ld in unique_ld]
    )  # (L, n_shared)

    if cs_cents.shape[0] >= 2:
        sim_cs = cosine_similarity(X_st_shared, cs_cents)
        margin_computed_mean = float(_margin(sim_cs).mean())
    else:
        margin_computed_mean = float("nan")

    if ld_cents.shape[0] >= 2:
        sim_ld = cosine_similarity(X_st_shared, ld_cents)
        margin_leiden_mean = float(_margin(sim_ld).mean())
    else:
        margin_leiden_mean = float("nan")

    # ── Weighted average perm_p (spot-weighted, mapped states only) ───────────
    total_spots = 0
    weighted_sum = 0.0
    for m in per_state.values():
        n_spots = m["n_spots"]
        p = m["perm_p_value"]
        if n_spots > 0 and p == p:  # n_spots >= 1 and p is not NaN
            weighted_sum += p * n_spots
            total_spots += n_spots
    weighted_perm_p = weighted_sum / total_spots if total_spots > 0 else float("nan")

    logger.info(
        "Substate analysis done: %d states  |  margin computed=%.4f  leiden=%.4f"
        "  |  weighted_perm_p=%.4f",
        len(per_state),
        margin_computed_mean,
        margin_leiden_mean,
        weighted_perm_p,
    )

    return {
        "per_state": per_state,
        "margin_computed_mean": margin_computed_mean,
        "margin_leiden_mean": margin_leiden_mean,
        "weighted_perm_p": weighted_perm_p,
    }
