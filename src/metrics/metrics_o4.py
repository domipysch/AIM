import enum
from typing import Any
import os
from pathlib import Path

import anndata
import pandas as pd
import networkx as nx
import numpy as np
from anndata import AnnData
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist
from .utils.dataset_query import get_z_real_and_predicted_data_only_shared_genes
import warnings
from .utils.distance_metrics import (
    cosine_similarity,
    sqrt_cosine_similarity,
    getis_ord_g_stat,
    pearson_distance,
    bray_curtis_distance,
    hellinger_distance,
    total_variation,
    bhattacharyya_distance,
    smape,
)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib import cm
import json
import logging

logger = logging.getLogger(__name__)


# ------------ Spatial Graph Creation


class NeighborhoodType(enum.Enum):
    KNN = "knn"
    DELAUNEY = "delauney"
    RADIUS = "radius"
    MUTUAL_KNN = "mutual_knn"
    RNG = "rng"  # Relative Neighborhood Graph


def create_spatial_graph(
    st_path: Path,
    neighborhood_type: NeighborhoodType = NeighborhoodType.KNN,
    k: int = 4,
    radius: float | None = None,
) -> nx.Graph:
    """
    Read st.h5ad and build a networkx.Graph with one node per spot (node attribute 'pos').

    Additional parameters:
    - neighborhood_type: NeighborhoodType (KNN, DELAUNEY, RADIUS, ...)
    - k: number of neighbors for KNN (default 4)
    - radius: radius for RADIUS mode (required when neighborhood_type == RADIUS)

    Edges get an attribute 'weight' with the euclidean distance.
    """

    adata_st = anndata.read_h5ad(st_path)

    # Get spot ids
    spot_ids = adata_st.obs_names.astype(str).tolist()

    # Get coordinates
    spatial = adata_st.obsm["spatial"]
    xs = spatial[:, 0].astype(float).tolist()
    ys = spatial[:, 1].astype(float).tolist()
    if not (len(spot_ids) == len(xs) == len(ys)):
        raise ValueError("Längen der Spot-IDs und Koordinaten stimmen nicht überein.")

    # Create graph
    G = nx.Graph()

    # Add nodes with positions
    for sid, x, y in zip(spot_ids, xs, ys):
        G.add_node(sid, pos=(float(x), float(y)))

    # If there are fewer than 2 nodes, nothing to connect
    n_nodes = len(spot_ids)
    coords = np.vstack([xs, ys]).T  # shape (n,2)
    tree = cKDTree(coords)

    if neighborhood_type == NeighborhoodType.KNN:
        assert k >= 1, "'k' muss mindestens 1 sein für KNN."
        # query returns self in first column if k+1 used
        dists, idxs = tree.query(coords, k=k + 1)
        for i, neighbors in enumerate(idxs):
            for j in neighbors:
                if j == i:
                    # No self-edges.
                    continue
                sid_i, sid_j = spot_ids[i], spot_ids[j]
                if not G.has_edge(sid_i, sid_j):
                    dist = float(np.linalg.norm(coords[i] - coords[j]))
                    G.add_edge(sid_i, sid_j, weight=dist)

    elif neighborhood_type == NeighborhoodType.MUTUAL_KNN:
        # Mutual kNN: add edge only if i in kNN(j) and j in kNN(i)
        assert k >= 1, "'k' muss mindestens 1 sein für MUTUAL_KNN."
        dists, idxs = tree.query(coords, k=k + 1)
        # build neighbor sets excluding self
        neigh_sets = []
        for i, neighbors in enumerate(idxs):
            nbrs = set(int(j) for j in neighbors if int(j) != i)
            neigh_sets.append(nbrs)
        # add mutual edges
        for i in range(n_nodes):
            for j in neigh_sets[i]:
                if j <= i:
                    continue  # add each edge once
                if i in neigh_sets[j]:
                    sid_i, sid_j = spot_ids[i], spot_ids[j]
                    if not G.has_edge(sid_i, sid_j):
                        dist = float(np.linalg.norm(coords[i] - coords[j]))
                        G.add_edge(sid_i, sid_j, weight=dist)

    elif neighborhood_type == NeighborhoodType.DELAUNEY:
        if n_nodes < 3:
            # can't triangulate
            return G
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            # add edges between each pair of simplex vertices
            for a, b in ((0, 1), (0, 2), (1, 2)):
                i = int(simplex[a])
                j = int(simplex[b])
                sid_i = spot_ids[i]
                sid_j = spot_ids[j]
                if not G.has_edge(sid_i, sid_j):
                    dist = float(np.linalg.norm(coords[i] - coords[j]))
                    G.add_edge(sid_i, sid_j, weight=dist)

    elif neighborhood_type == NeighborhoodType.RNG:
        # Relative Neighborhood Graph (RNG)
        # Use Delaunay to get candidate edges (RNG is a subgraph of Delaunay)
        if n_nodes < 3:
            # With only two points, the single edge is RNG
            sid0, sid1 = spot_ids[0], spot_ids[1]
            dist01 = float(np.linalg.norm(coords[0] - coords[1]))
            G.add_edge(sid0, sid1, weight=dist01)
            return G
        tri = Delaunay(coords)
        candidate_edges = set()
        for simplex in tri.simplices:
            verts = [int(v) for v in simplex]
            for a, b in ((0, 1), (0, 2), (1, 2)):
                i = verts[a]
                j = verts[b]
                if i == j:
                    continue
                a_idx, b_idx = min(i, j), max(i, j)
                candidate_edges.add((a_idx, b_idx))
        eps = 1e-10
        for i, j in candidate_edges:
            pij = np.linalg.norm(coords[i] - coords[j])
            # distances from i and j to all points
            d_i = np.linalg.norm(coords - coords[i], axis=1)
            d_j = np.linalg.norm(coords - coords[j], axis=1)
            # exclude i and j
            mask = np.ones(n_nodes, dtype=bool)
            mask[i] = False
            mask[j] = False
            # if any k exists with max(d_i[k], d_j[k]) < pij then (i,j) is NOT in RNG
            if np.any(np.maximum(d_i[mask], d_j[mask]) < (pij - eps)):
                continue
            sid_i, sid_j = spot_ids[i], spot_ids[j]
            if not G.has_edge(sid_i, sid_j):
                G.add_edge(sid_i, sid_j, weight=float(pij))

    elif neighborhood_type == NeighborhoodType.RADIUS:
        # Check argument
        if radius is None:
            raise ValueError("Für RADIUS-Modus muss 'radius' angegeben werden.")
        neighbors_list = tree.query_ball_point(coords, r=radius)
        for i, nbrs in enumerate(neighbors_list):
            for j in nbrs:
                if i == j:
                    continue
                if j < i:
                    continue
                sid_i = spot_ids[i]
                sid_j = spot_ids[j]
                if not G.has_edge(sid_i, sid_j):
                    dist = float(np.linalg.norm(coords[i] - coords[j]))
                    G.add_edge(sid_i, sid_j, weight=dist)

    else:
        raise ValueError(f"Unbekannter neighborhood_type: {neighborhood_type}")

    # remove any accidental self-loops for safety before returning
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


# In Tangram Refined: L_
def binary_adjacency_matrix_from_graph(st_path: Path, G: nx.Graph) -> pd.DataFrame:
    """
    Create a binary (0/1) adjacency matrix as a pandas.DataFrame from the given networkx Graph G.
    The row/column order matches exactly the order of spots in st.h5ad.

    Returns:
    - A: pandas.DataFrame, shape (n_spots, n_spots), values 0/1, Index/Columns = spot_ids
    """
    adata_st = anndata.read_h5ad(st_path)
    spot_ids = adata_st.obs_names.astype(str).tolist()
    n = len(spot_ids)
    idx_map = {sid: i for i, sid in enumerate(spot_ids)}

    A = np.zeros((n, n), dtype=np.int8)

    # iterate edges and fill matrix; ignore edges whose nodes are not in spot list
    for u, v in G.edges():
        su, sv = str(u), str(v)
        if su not in idx_map or sv not in idx_map:
            warnings.warn(
                f"Kante ({su},{sv}) weist auf Spot-ID, die nicht in st.h5ad enthalten ist. Ignoriere."
            )
            continue
        i = idx_map[su]
        j = idx_map[sv]
        A[i, j] = 1
        A[j, i] = 1

    # Assert diagonal entries are already zero (no unexpected self-loops)
    assert np.all(
        np.diag(A) == 0
    ), f"Adjazenz-Matrix enthält nicht-null Diagonaleinträge"

    # Return as pandas DataFrame with spot ids as index/columns
    return pd.DataFrame(A, index=spot_ids, columns=spot_ids)


# In Tangram Refined: L
def locality_matrix(
    st_path: Path,
    method: str = "rbf",  # "rbf", "linear", "inverse"
    sigma: float | None = None,  # used for 'rbf'; if None inferred from data
    dtype=np.float32,
) -> pd.DataFrame:
    """
    Create a symmetric neighborhood/similarity matrix as a pandas.DataFrame
    (n x n, values in [0,1]) for the spots in st.h5ad. The row/column
    ordering matches exactly the order of spots.

    Returns: pandas.DataFrame with Index/Columns = spot_ids (as strings) and dtype dtype.
    """

    adata_st = anndata.read_h5ad(st_path)
    spot_ids = adata_st.obs_names.astype(str).tolist()
    n = len(spot_ids)

    spatial = adata_st.obsm["spatial"]
    coords = spatial.astype(float)

    # Build distance matrix
    D = cdist(coords, coords, metric="euclidean")

    # Convert distances to similarities in [0,1]
    sim = np.zeros_like(D, dtype=float)

    # handle trivial case n==1
    if n == 1:
        return pd.DataFrame([[1.0]], index=spot_ids, columns=spot_ids, dtype=dtype)

    if method == "rbf":
        # infer sigma if not given: median of nearest-neighbor distances (exclude zeros on diagonal)
        if sigma is None:
            # for each row, find smallest positive distance
            nn = np.partition(D, 1, axis=1)[:, 1]
            finite_nn = nn[np.isfinite(nn) & (nn > 0)]
            if finite_nn.size == 0:
                sigma = 1.0
            else:
                sigma = float(np.median(finite_nn))
                if sigma <= 0:
                    sigma = 1.0
        sigma = float(sigma)
        denom = 2.0 * (sigma**2)
        # avoid overflow: where D is inf -> similarity 0
        with np.errstate(over="ignore"):
            sim = np.exp(-(D**2) / denom)
        sim[~np.isfinite(sim)] = 0.0

    elif method == "linear":
        # normalize by max finite distance
        finite_mask = np.isfinite(D)
        if not np.any(finite_mask):
            maxd = 1.0
        else:
            maxd = float(np.max(D[finite_mask]))
            if maxd == 0:
                maxd = 1.0
        sim = 1.0 - (D / maxd)
        sim[~np.isfinite(sim)] = 0.0
        sim = np.clip(sim, 0.0, 1.0)

    elif method == "inverse":
        # 1 / (1 + d) maps [0, inf) -> (0,1], diag=1
        sim = 1.0 / (1.0 + D)
        sim[~np.isfinite(sim)] = 0.0
    else:
        raise ValueError("method must be one of 'rbf', 'linear', 'inverse'")

    # Ensure symmetry (average numerical noise) and diagonal ones
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 1.0)

    # Convert to pandas DataFrame with spot ids as index/columns and proper dtype
    return pd.DataFrame(sim, index=spot_ids, columns=spot_ids)


def compute_tangram_refined_metric_1(
    adata_z: AnnData, adata_predicted_z: AnnData, dataset_folder: Path
) -> pd.DataFrame:
    """
    Compute for each spot (in the order adata_z.obs_names) two metrics:
    - cossim: cosine similarity between the (locally smoothed) observed and predicted vector (over genes)
    - sqrt_cossim: sqrt-cosine similarity

    Returns: pandas.DataFrame with Index = spot ids and columns ['cossim', 'sqrt_cossim'].
    The AnnData objects are not modified.
    """
    # Get locality matrix (pandas DataFrame with spot_ids index/columns)
    lm = locality_matrix(dataset_folder)
    if lm.shape[0] != adata_z.n_obs:
        raise AssertionError(
            "Locality matrix und AnnData haben unterschiedliche Anzahl an Spots."
        )

    # Reorder locality matrix to match adata row order (adata.obs_names must match spot ids)
    lm_ordered = lm.loc[
        adata_z.obs_names, adata_z.obs_names
    ].values  # numpy array in correct order

    # helper to get dense array
    def _dense(X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    # Apply locality matrix without modifying the AnnData objects
    local_z = lm_ordered.dot(_dense(adata_z.X))  # shape (n_spots, n_genes)
    local_pred = lm_ordered.dot(_dense(adata_predicted_z.X))  # same shape

    n_spots = local_z.shape[0]
    spot_ids = list(adata_z.obs_names)

    cossim_vals = np.empty(n_spots, dtype=float)
    sqrt_cossim_vals = np.empty(n_spots, dtype=float)

    for i in range(n_spots):
        vec_z = np.asarray(local_z[i, :], dtype=float).ravel()
        vec_pred = np.asarray(local_pred[i, :], dtype=float).ravel()
        cossim_vals[i] = float(cosine_similarity(vec_z, vec_pred))
        sqrt_cossim_vals[i] = float(sqrt_cosine_similarity(vec_z, vec_pred))

    result_df = pd.DataFrame(
        {"cossim": cossim_vals, "sqrt_cossim": sqrt_cossim_vals}, index=spot_ids
    )

    return result_df


def compute_tangram_refined_metric_3(
    adata_z: AnnData,
    adata_predicted_z: AnnData,
    dataset_folder: Path,
    save_gog_json: Path | None = None,
) -> pd.DataFrame:
    """
    Compute the Getis-Ord G* statistic for each spot (in the order adata_z.obs_names).

    Returns: pandas.DataFrame with Index = spot ids and column ['gog'].
    The AnnData objects are not modified.
    """

    # helper to get dense array
    def _dense(X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    # --- read locality matrix (DataFrame) and convert to numpy for repeated use ---
    blm = binary_adjacency_matrix_from_graph(
        dataset_folder,
        create_spatial_graph(
            dataset_folder, neighborhood_type=NeighborhoodType.KNN, k=4
        ),
    )
    W = blm.values
    n_genes = adata_z.n_vars

    # dense data matrices: shape (n_spots, n_genes)
    data_z = _dense(adata_z.X)
    data_pred = _dense(adata_predicted_z.X)

    # For each gene compute G* across spots; result matrices shape (n_spots, n_genes)
    gene_ids = list(adata_z.var_names)
    gog_vals = np.empty(n_genes, dtype=float)

    # Collect cossim-like values for optional export
    gog_dict = {}

    counter = 0
    for g in range(n_genes):

        getis_z = getis_ord_g_stat(data_z[:, g], W)
        getis_z_pred = getis_ord_g_stat(data_pred[:, g], W)

        if np.any(np.isnan(getis_z)) or np.any(np.isnan(getis_z_pred)):
            gog_vals[g] = float("nan")
        else:
            value = float(cosine_similarity(getis_z, getis_z_pred))
            gog_vals[g] = value
            gog_dict[gene_ids[g]] = value

        counter += 1
        if counter % 2000 == 0:
            logging.info(f"Fertig mit {counter}/{n_genes} Genen für TG Ref Metric 3")

    result_df = pd.DataFrame(
        {
            "gog": gog_vals,
        },
        index=gene_ids,
    )

    # Optional: save gog per gene as json
    if save_gog_json is not None:
        save_gog_json.parent.mkdir(parents=True, exist_ok=True)
        # JSON-serializable (floats already)
        with save_gog_json.open("w", encoding="utf-8") as f:
            json.dump(gog_dict, f, indent=4)

    return result_df


def visualize_tangram_refined_metrics(
    metric_1_df: pd.DataFrame,
    metric_3_df: pd.DataFrame,
    figsize: tuple = (10, 4),
    output_folder: Path | None = None,
) -> None:
    """
    Show two subplots side-by-side:
    - left: two boxplots for 'cossim' and 'sqrt_cossim' from metric_1_df
    - right: one boxplot for 'gog' from metric_3_df

    No files are saved by default; plt.show() is called if output_folder is None.
    """
    # required columns
    required_m1 = ["cossim", "sqrt_cossim"]
    for col in required_m1:
        if col not in metric_1_df.columns:
            warnings.warn(
                f"metric_1_df enthält nicht die erwartete Spalte '{col}'. Abbruch."
            )
            return
    if "gog" not in metric_3_df.columns:
        warnings.warn("metric_3_df enthält nicht die erwartete Spalte 'gog'. Abbruch.")
        return

    # extract and clean data
    s_cossim = metric_1_df["cossim"].dropna().astype(float).values
    s_sqrt = metric_1_df["sqrt_cossim"].dropna().astype(float).values
    s_gog = metric_3_df["gog"].dropna().astype(float).values

    if s_cossim.size == 0 and s_sqrt.size == 0:
        warnings.warn("Keine Werte für 'cossim' oder 'sqrt_cossim' gefunden. Abbruch.")
        return
    if s_gog.size == 0:
        warnings.warn("Keine Werte für 'gog' gefunden. Abbruch.")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Left: cossim & sqrt_cossim as two boxes
    data_left = []
    labels_left = []
    if s_cossim.size > 0:
        data_left.append(s_cossim)
        labels_left.append("cossim")
    if s_sqrt.size > 0:
        data_left.append(s_sqrt)
        labels_left.append("sqrt_cossim")

    bp_left = axes[0].boxplot(
        data_left, labels=labels_left, patch_artist=True, notch=False
    )
    colors_left = cm.get_cmap("viridis")(np.linspace(0.25, 0.6, len(data_left)))
    for patch, color in zip(bp_left.get("boxes", []), colors_left):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[0].set_title("Metric 1: cossim & sqrt_cossim")
    axes[0].set_xticks(range(1, len(labels_left) + 1))
    # annotate counts
    for i, arr in enumerate(data_left, start=1):
        axes[0].text(
            i,
            axes[0].get_ylim()[0],
            f"n={len(arr)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    # Right: gog single boxplot
    bp_right = axes[1].boxplot([s_gog], labels=["gog"], patch_artist=True, notch=False)
    for patch in bp_right.get("boxes", []):
        patch.set_facecolor(cm.get_cmap("viridis")(0.85))
        patch.set_alpha(0.8)
    axes[1].set_title("Metric 3: gog")
    axes[1].set_xticks([1])
    axes[1].text(
        1,
        axes[1].get_ylim()[0],
        f"n={s_gog.size}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    axes[0].set_ylabel("Wert")
    plt.tight_layout()

    if output_folder:
        plt.savefig(output_folder / f"o4_tg_ref_metrics.pdf", bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


# ------------ Own locality Metrics


def add_own_metrics_to_edges(
    adata_z: AnnData,
    adata_predicted_z: AnnData,
    graph: nx.Graph,
    save_own_cossim: Path | None = None,
) -> nx.Graph:

    # helper to get dense 1D numpy array for a given spot id (obs_name)
    def _get_spot_vector(adata: AnnData, spot_id: str) -> np.ndarray:
        row_idx = int(np.where(np.asarray(adata.obs_names) == spot_id)[0][0])
        row = adata.X[row_idx]
        return (
            (row.toarray() if hasattr(row, "toarray") else np.asarray(row))
            .ravel()
            .astype(float)
        )

    # metrics to compute (name, function)
    metrics = [
        ("cossim", cosine_similarity),
        # ("sqrt_cossim", sqrt_cosine_similarity),
        # ("pearson", pearson_distance),
        # ("bray_curtis", bray_curtis_distance),
        # ("hellinger", hellinger_distance),
        # ("bhat", bhattacharyya_distance),
        # ("tv", total_variation),
        # ("smape", smape),
    ]

    cossim_dict = {}

    # iterate over edges and annotate
    for u, v in graph.edges():
        su, sv = str(u), str(v)

        vec_u_z = _get_spot_vector(adata_z, su)
        vec_v_z = _get_spot_vector(adata_z, sv)
        vec_u_pred = _get_spot_vector(adata_predicted_z, su)
        vec_v_pred = _get_spot_vector(adata_predicted_z, sv)

        # compute each metric for z and pred, store absolute diff
        for name, func in metrics:

            val_z = float(func(vec_u_z, vec_v_z))
            val_pred = float(func(vec_u_pred, vec_v_pred))
            diff = float(abs(val_z - val_pred))

            # graph[u][v][f"{name}_z"] = val_z
            # graph[u][v][f"{name}_pred"] = val_pred
            graph[u][v][f"{name}_diff"] = diff

            if name == "cossim":
                # Collect for export
                cossim_dict[f"{su}-{sv}"] = diff

    # Optional: save cossim per edge as json
    if save_own_cossim is not None:
        save_own_cossim.parent.mkdir(parents=True, exist_ok=True)
        # JSON-serializable (floats already)
        with save_own_cossim.open("w", encoding="utf-8") as f:
            json.dump(cossim_dict, f, indent=4)

    return graph


def create_box_plots_from_edge_annots(
    graph: nx.Graph,
    metrics: list[Any] | None = None,
    figsize: tuple = (14, 6),
    output_folder: Path | None = None,
) -> None:
    """
    Create side-by-side boxplots for all specified metrics based on the
    edge annotations in the graph. For each metric the edge attribute name
    "{metric}_diff" is used and all existing values are aggregated into a boxplot.

    Parameters:
    - graph: networkx.Graph, expects edge attributes like "cossim_diff", "pearson_diff", ...
    - metrics: list of metric names (without "_diff"). If None, a default list is used.
    - figsize: tuple for figure size.
    - output_folder: when provided, PNG files are written there; otherwise plots are shown.
    """
    if metrics is None:
        metrics = [
            "cossim",
            "sqrt_cossim",
            "pearson",
            "bray_curtis",
            "hellinger",
            "bhat",
            "tv",
            "smape",
        ]

    data_list = []
    labels = []
    counts = []

    # gather values per metric
    for m in metrics:
        key = f"{m}_diff"
        vals = []
        for _, _, d in graph.edges(data=True):
            if key in d:
                v = d.get(key)
                try:
                    # only finite numeric values
                    if v is None:
                        continue
                    fv = float(v)
                    if np.isfinite(fv):
                        vals.append(fv)
                except Exception:
                    continue
        if len(vals) == 0:
            # skip metrics without values
            continue
        data_list.append(vals)
        labels.append(m)
        counts.append(len(vals))

    if len(data_list) == 0:
        warnings.warn("Keine Metrik-Werte in Graph gefunden. Keine Boxplots erzeugt.")
        return

    # create figure and boxplots
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True, notch=False)

    # color boxes
    for patch, color in zip(
        bp["boxes"], plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(bp["boxes"])))
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Metrik")
    ax.set_ylabel("Wert (|z - pred| oder metric-specific)")
    ax.set_title("Edge-wise Metrik-Vergleich (Boxplots)")

    # annotate counts under each box
    xticks = np.arange(1, len(labels) + 1)
    for xt, cnt in zip(xticks, counts):
        ax.text(
            float(xt),
            ax.get_ylim()[0],
            f"n={cnt}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
            rotation=0,
        )

    plt.tight_layout()
    if output_folder:
        plt.savefig(output_folder / f"o4_edges_box_plots.pdf", bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


# --- newly added: spatial visualization of edge metrics (e.g. cossim_diff) ---
def plot_edge_cossim_spatial(
    G: nx.Graph,
    metric: str = "cossim_diff",
    cmap: str = "viridis",
    node_size: int = 20,
    edge_min_width: float = 0.5,
    edge_max_width: float = 5.0,
    output_folder: Path | None = None,
):
    """
    Draw the graph edges spatially (based on node attribute 'pos') and color/scale
    them according to the edge attribute `metric` (e.g. "cossim_diff").

    Parameters:
    - G: networkx.Graph with node attribute 'pos' = (x,y) and edge attributes like "cossim_diff"
    - metric: edge attribute name used for color/width
    - cmap: matplotlib colormap name
    - node_size: size of nodes (spots)
    - edge_min_width / edge_max_width: min/max line width for edges
    - output_folder: when provided, the plot is saved; otherwise displayed
    """
    # collect positions
    pos = nx.get_node_attributes(G, "pos")
    if len(pos) == 0:
        warnings.warn("Keine 'pos' Attribute in Graph-Nodes gefunden. Abbruch.")
        return

    lines = []
    vals: Any = []
    for u, v, d in G.edges(data=True):
        if metric not in d:
            continue
        if u not in pos or v not in pos:
            continue
        p1 = pos[u]
        p2 = pos[v]
        lines.append([p1, p2])
        try:
            vals.append(float(d[metric]))
        except Exception:
            vals.append(np.nan)

    if len(lines) == 0:
        warnings.warn(
            f"Keine Kanten mit Attribut '{metric}' gefunden. Nichts zu plotten."
        )
        return

    vals = np.array(vals, dtype=float)
    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        warnings.warn(
            f"Alle Werte für '{metric}' sind nicht-finite. Nichts zu plotten."
        )
        return

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    # avoid zero-range
    if np.isclose(vmin, vmax):
        # expand tiny range for color mapping
        eps = 1e-8
        vmin = vmin - eps
        vmax = vmax + eps

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)
    colors = cmap_obj(norm(vals))
    # scale widths according to normalized value in [0,1]
    scaled = norm(vals)
    widths = edge_min_width + (edge_max_width - edge_min_width) * scaled
    # line collection
    lc = LineCollection(lines, colors=colors, linewidths=widths, zorder=1, alpha=0.9)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(lc)

    # draw nodes
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    ax.scatter(xs, ys, s=node_size, c="black", zorder=2)

    # set limits with some padding
    all_x = np.array(xs)
    all_y = np.array(ys)
    if all_x.size and all_y.size:
        xpad = (all_x.max() - all_x.min()) * 0.05 if all_x.max() != all_x.min() else 1.0
        ypad = (all_y.max() - all_y.min()) * 0.05 if all_y.max() != all_y.min() else 1.0
        ax.set_xlim(all_x.min() - xpad, all_x.max() + xpad)
        ax.set_ylim(all_y.min() - ypad, all_y.max() + ypad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Edge spatial plot — metric: {metric}")

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array(vals)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)

    ax.axis("off")
    plt.tight_layout()

    if output_folder:
        plt.savefig(output_folder / f"o4_edges_spatial_plots.pdf", bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def main(
    sc_path: Path, st_path: Path, result_gep: AnnData, metrics_output_folder: Path
):
    """
    Compute metrics for objective 4 and save results as JSON files / diagrams.

    - Compute spatial graph out of ST data
    - Compute Tangram Refined metrics 1 and 3
    - Compute own locality-based metrics on graph edges

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        result_gep: G x S
        metrics_output_folder:

    Returns: None
    """
    logger.info("Compute metrics for o4")

    # Assemble file and folder names
    output_folder = metrics_output_folder / "o4"
    os.makedirs(output_folder, exist_ok=True)
    output_folder_knn = output_folder / "knn"
    os.makedirs(output_folder_knn, exist_ok=True)
    output_folder_delauney = output_folder / "delauney"
    os.makedirs(output_folder_delauney, exist_ok=True)

    result_own_knn_cossim_file = output_folder_knn / "cossim.json"

    # S x shared G
    adata_z, adata_predicted_z = get_z_real_and_predicted_data_only_shared_genes(
        sc_path, st_path, result_gep
    )
    # Assert that both DataFrames have the same shape of genes and spots
    assert (
        adata_z.shape == adata_predicted_z.shape
    ), "DataFrames haben unterschiedliche Formen."
    assert adata_z.n_obs == result_gep.n_vars

    # KNN
    graph = create_spatial_graph(st_path, neighborhood_type=NeighborhoodType.KNN, k=4)
    graph = add_own_metrics_to_edges(
        adata_z,
        adata_predicted_z,
        graph,
        save_own_cossim=result_own_knn_cossim_file,
    )
    create_box_plots_from_edge_annots(
        graph,
        output_folder=output_folder_knn,
    )
