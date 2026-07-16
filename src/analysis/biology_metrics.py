"""Biological / topological metrics for AIM's decoupled post-mapping analysis.

Two questions, each producing scalar objectives suitable for a grid search
(every metric is reported with an observed value AND a permutation-null
z-score; the z-score is the dataset-comparable objective, since raw purity /
Moran's I / cosine values are not comparable across pairs with different
numbers of states):

1. Spatial organisation of the mapped spots (P's argmax state per spot)
   - Local Spatial Purity (LSP): mean fraction of the k nearest spatial
     neighbours sharing the same mapped state — sensitive to local compactness.
   - Moran's I averaged over the per-state binary indicators — sensitive to
     global spatial clustering (via squidpy).
   - Null: shuffle the spot-state labels N_PERM times.

2. Coherence of the Leiden clusters AIM merged into one computed state
   - For each computed state that aggregates >=2 Leiden clusters: are those
     clusters' shared-gene centroids MORE mutually similar (higher mean pairwise
     cosine similarity) than a same-sized random draw of Leiden clusters
     belonging to OTHER states? If yes (high z-score, low p), the merge looks
     coherent in the gene subspace AIM actually operates in.
   - Restricted to the sc/ST shared genes (the space AIM sees ST through).
   - Null: N_PERM random same-sized draws of Leiden clusters from other states.

Unlike the 00_Playground prototypes this is adapted from, the Leiden -> state
grouping is NOT recomputed / majority-voted here: in the current two-level model
it is exactly the argmax of AIM's merge matrix G (leiden_merge_prob.h5ad), so it
is read directly from G_hard.

Classifier-based substate *separability* is intentionally not computed here (too
expensive to run once per grid-search config).
"""

from __future__ import annotations

import logging
from typing import Callable

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
K_SPATIAL = 6  # neighbours for local spatial purity / Moran's I KNN graph
N_PERM_SPATIAL = 200  # permutations for the spatial-organisation null
N_PERM_COHERENCE = 200  # permutations for the substate-coherence null
SIG_ALPHA = 0.05  # p-value threshold for "significant" counts


# ── Permutation test ──────────────────────────────────────────────────────────
def permutation_test(
    observed: float,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    """Shuffle ``labels`` n_perm times, recompute ``metric_fn`` each time.

    Returns observed value plus p-value (fraction of null >= observed), z-score,
    and the null mean/std. Full null draws are not retained (keeps the JSON small).
    """
    if not np.isfinite(observed):
        return {
            "observed": float(observed),
            "p_value": float("nan"),
            "z_score": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
        }
    perm = np.array([metric_fn(rng.permutation(labels)) for _ in range(n_perm)])
    valid = perm[np.isfinite(perm)]
    if len(valid) == 0:
        return {
            "observed": float(observed),
            "p_value": float("nan"),
            "z_score": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
        }
    return {
        "observed": float(observed),
        "p_value": float((valid >= observed).mean()),
        "z_score": float((observed - valid.mean()) / (valid.std() + 1e-12)),
        "null_mean": float(valid.mean()),
        "null_std": float(valid.std()),
    }


# ── 1. Spatial organisation of mapped spots ───────────────────────────────────
def _knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """(n_spots, k) indices of each spot's k nearest spatial neighbours (self excluded)."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(coords) - 1)
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, idx = knn.kneighbors(coords)
    return idx[:, 1:]


def local_spatial_purity(labels: np.ndarray, nbr_idx: np.ndarray) -> float:
    """Mean fraction of k nearest spatial neighbours sharing the same state label,
    using precomputed neighbour indices (so the permutation null is cheap)."""
    return float((labels[nbr_idx] == labels[:, None]).mean())


def _spatial_neighbors_graph(coords: np.ndarray, k: int) -> ad.AnnData:
    """Minimal AnnData holding a squidpy-built spatial KNN graph, reused across permutations."""
    import squidpy as sq

    graph = ad.AnnData(
        X=np.zeros((len(coords), 1), dtype=np.float32),
        obsm={"spatial": np.asarray(coords, dtype=np.float32)},
    )
    sq.gr.spatial_neighbors(
        graph, n_neighs=min(k, len(coords) - 1), coord_type="generic"
    )
    return graph


def morans_i_mean(labels: np.ndarray, graph: ad.AnnData) -> float:
    """Moran's I averaged over all states' binary indicators, via squidpy.gr.spatial_autocorr."""
    import squidpy as sq

    states = np.unique(labels)
    if len(states) < 2:
        return float("nan")
    state_cols = [f"state_{s}" for s in states]
    onehot = np.zeros((len(labels), len(state_cols)), dtype=np.float32)
    for col, s in enumerate(states):
        onehot[:, col] = labels == s
    graph.obs[state_cols] = onehot

    result = sq.gr.spatial_autocorr(
        graph,
        mode="moran",
        genes=state_cols,
        attr="obs",
        connectivity_key="spatial_connectivities",
        n_perms=None,
        n_jobs=1,
        show_progress_bar=False,
        copy=True,
    )
    values = result["I"].to_numpy()
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else float("nan")


def compute_spatial_organization(
    spot_states: np.ndarray,
    coords: np.ndarray | None,
    k: int = K_SPATIAL,
    n_perm: int = N_PERM_SPATIAL,
    seed: int = 0,
) -> dict:
    """Local spatial purity + mean Moran's I of the mapped spot states, each with
    a permutation-null z-score. Metrics that cannot be computed (no coordinates,
    <2 states, squidpy unavailable) are returned as NaN rather than raising.
    """
    out: dict = {
        "n_spots": int(len(spot_states)),
        "n_mapped_states": int(len(np.unique(spot_states))),
        "k": int(k),
        "n_perm": int(n_perm),
        "local_purity": None,
        "morans_i": None,
    }
    if coords is None or len(coords) != len(spot_states):
        logger.warning(
            "Spatial organisation skipped: coordinates missing or length mismatch "
            "(coords=%s, spots=%d).",
            None if coords is None else len(coords),
            len(spot_states),
        )
        return out
    if len(spot_states) <= k:
        logger.warning(
            "Spatial organisation skipped: too few spots (%d).", len(spot_states)
        )
        return out

    rng = np.random.default_rng(seed)

    nbr_idx = _knn_indices(coords, k)
    obs_lsp = local_spatial_purity(spot_states, nbr_idx)
    out["local_purity"] = permutation_test(
        obs_lsp, spot_states, lambda l: local_spatial_purity(l, nbr_idx), n_perm, rng
    )

    try:
        graph = _spatial_neighbors_graph(coords, k)
        obs_mi = morans_i_mean(spot_states, graph)
        out["morans_i"] = permutation_test(
            obs_mi, spot_states, lambda l: morans_i_mean(l, graph), n_perm, rng
        )
    except Exception as e:  # squidpy missing or autocorr failure — degrade gracefully
        logger.warning("Moran's I skipped (%s: %s).", type(e).__name__, e)

    return out


# ── 2. Substate merge coherence ────────────────────────────────────────────────
def leiden_state_groups(G_hard: np.ndarray) -> dict[int, list[int]]:
    """Map each computed state to the list of Leiden clusters hard-merged into it.

    G_hard is the argmax one-hot of AIM's merge matrix G (L_leiden x L_states);
    state of Leiden cluster l is argmax(G_hard[l]). This is the exact grouping the
    model learned — no recomputation / majority vote (as the prototypes needed).
    """
    state_of_leiden = np.asarray(G_hard).argmax(axis=1)
    groups: dict[int, list[int]] = {}
    for leiden_cluster, state in enumerate(state_of_leiden):
        groups.setdefault(int(state), []).append(int(leiden_cluster))
    return groups


def _pairwise_cosine_stats(vecs: np.ndarray) -> tuple[float, float]:
    """(mean, median) pairwise cosine similarity over all unique row pairs of ``vecs``."""
    n = len(vecs)
    if n < 2:
        return 1.0, 1.0
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / (norm + 1e-12)
    sim = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    sims = sim[iu]
    return float(sims.mean()), float(np.median(sims))


def compute_substate_coherence(
    centroids: np.ndarray,
    G_hard: np.ndarray,
    n_perm: int = N_PERM_COHERENCE,
    seed: int = 0,
) -> dict:
    """Shared-gene centroid coherence of the Leiden clusters merged into each state.

    For every computed state that aggregates >=2 Leiden clusters (and with at
    least one Leiden cluster belonging to another state, to draw a null from):
    compute the mean/median pairwise cosine similarity of the merged clusters'
    shared-gene centroids, and permutation-test it against ``n_perm`` random
    same-sized draws of Leiden clusters from OTHER states.

    Args:
        centroids: per-Leiden-cluster shared-gene centroid (L x G_shared), e.g.
                   normalized+log1p expression sums / sizes. Rows for empty Leiden
                   clusters (size 0) may be all-zero; such clusters are dropped.
        G_hard:    argmax one-hot merge matrix (L x L); defines the grouping.

    Returns a dict with per-state detail and aggregate scalars.
    """
    groups = leiden_state_groups(G_hard)
    # Leiden clusters that actually have cells (non-zero centroid) — a zero
    # centroid has undefined cosine similarity and must not seed a null draw.
    valid_leiden = {
        int(l) for l in range(len(centroids)) if np.any(centroids[l] != 0.0)
    }

    per_state: dict[int, dict] = {}
    for state, leiden_clusters in sorted(groups.items()):
        targets = [l for l in leiden_clusters if l in valid_leiden]
        others = [l for l in valid_leiden if l not in leiden_clusters]
        if len(targets) < 2:
            per_state[state] = {
                "n_leiden_sub": len(targets),
                "skipped_reason": "fewer than 2 non-empty merged Leiden clusters",
            }
            continue
        if not others:
            per_state[state] = {
                "n_leiden_sub": len(targets),
                "skipped_reason": "no Leiden clusters in other states to draw a null from",
            }
            continue

        rng = np.random.default_rng(seed + state)
        obs_mean, obs_median = _pairwise_cosine_stats(centroids[targets])

        n_draw = len(targets)
        replace = n_draw > len(others)
        others_arr = np.array(others)
        null_means = np.empty(n_perm)
        for i in range(n_perm):
            draw = rng.choice(others_arr, size=n_draw, replace=replace)
            null_means[i], _ = _pairwise_cosine_stats(centroids[draw])

        p_mean = float((null_means >= obs_mean).mean())
        z_mean = float((obs_mean - null_means.mean()) / (null_means.std() + 1e-12))
        per_state[state] = {
            "n_leiden_sub": len(targets),
            "mean_cossim": obs_mean,
            "median_cossim": obs_median,
            "null_mean": float(null_means.mean()),
            "p_value_mean": p_mean,
            "z_score_mean": z_mean,
            "skipped_reason": None,
        }

    tested = [m for m in per_state.values() if m.get("skipped_reason") is None]
    if tested:
        aggregate = {
            "n_tested_states": len(tested),
            "mean_cossim": float(np.mean([m["mean_cossim"] for m in tested])),
            "mean_z_score": float(np.mean([m["z_score_mean"] for m in tested])),
            "frac_significant": float(
                np.mean([m["p_value_mean"] < SIG_ALPHA for m in tested])
            ),
        }
    else:
        aggregate = {
            "n_tested_states": 0,
            "mean_cossim": float("nan"),
            "mean_z_score": float("nan"),
            "frac_significant": float("nan"),
        }

    return {"n_perm": int(n_perm), "aggregate": aggregate, "per_state": per_state}


# ── Flat objective scalars ─────────────────────────────────────────────────────
def flatten_biology_objectives(spatial: dict, coherence: dict) -> dict[str, float]:
    """Flatten the spatial + coherence dicts into the scalar objectives a grid
    search consumes (prefixed so they never collide with loss-term columns)."""

    def _g(metric: dict | None, key: str) -> float:
        return float(metric[key]) if metric else float("nan")

    lsp = spatial.get("local_purity")
    mi = spatial.get("morans_i")
    agg = coherence["aggregate"]
    return {
        "bio_spatial_local_purity": _g(lsp, "observed"),
        "bio_spatial_local_purity_z": _g(lsp, "z_score"),
        "bio_spatial_morans_i": _g(mi, "observed"),
        "bio_spatial_morans_i_z": _g(mi, "z_score"),
        "bio_coherence_mean_cossim": float(agg["mean_cossim"]),
        "bio_coherence_mean_z": float(agg["mean_z_score"]),
        "bio_coherence_frac_significant": float(agg["frac_significant"]),
        "bio_coherence_n_tested_states": float(agg["n_tested_states"]),
    }
