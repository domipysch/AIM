from typing import Optional

from anndata import AnnData
import torch
from enum import Enum, auto
from scipy.spatial import KDTree, Delaunay
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class SpatialGraphType(Enum):
    """
    Topology options for the spatial neighborhood graph.

    KNN         — directed k-nearest-neighbor graph (each spot has exactly k outgoing edges).
    MUTUAL_KNN  — undirected mutual kNN (edge (i,j) only if j ∈ kNN(i) AND i ∈ kNN(j)).
    RADIUS      — epsilon-ball graph (all spots within a fixed radius are connected).
    DELAUNAY    — Delaunay triangulation (adapts naturally to varying spot density).
    """

    KNN = auto()
    MUTUAL_KNN = auto()
    RADIUS = auto()
    DELAUNAY = auto()


def build_spatial_graph(
    adata_st: AnnData,
    method: SpatialGraphType = SpatialGraphType.KNN,
    k: Optional[int] = 4,
    radius: Optional[float] = None,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Builds the spatial graph g from spot coordinates using various topologies.

    Args:
        adata_st: Spatial AnnData object containing coordinates in obsm['spatial'].
        method: SpatialGraphType.
        k: Number of neighbors (for KNN and MUTUAL_KNN).
        radius: Distance threshold (for RADIUS).
        device: Target device for the tensor (e.g., 'cpu', 'cuda', 'mps').

    Returns:
        edge_index: Tensor of shape (2, E) in PyTorch Geometric format.
    """
    logger.debug(f"Build spatial graph of type {method}")
    assert (
        adata_st.obsm["spatial"] is not None
    ), "No spatial coordinates found in adata_st.obsm['spatial']"
    coords = adata_st.obsm["spatial"]
    num_spots = coords.shape[0]
    tree = KDTree(coords)

    edge_list = []

    if method == SpatialGraphType.KNN:
        assert k is not None
        # Standard kNN graph
        _, indices = tree.query(coords, k=k + 1)
        for i in range(num_spots):
            for neighbor in indices[i]:
                if i != neighbor:
                    edge_list.append((i, neighbor))

    elif method == SpatialGraphType.MUTUAL_KNN:
        assert k is not None
        # i and j are connected only if i is in kNN(j) AND j is in kNN(i)
        _, indices = tree.query(coords, k=k + 1)
        # Create a list of sets for fast lookup
        adj_sets = [set(indices[i]) - {i} for i in range(num_spots)]
        for i in range(num_spots):
            for j in adj_sets[i]:
                if i in adj_sets[j]:
                    edge_list.append((i, j))

    elif method == SpatialGraphType.RADIUS:
        # Fixed distance epsilon-graph
        if radius is None:
            raise ValueError("Radius must be specified for SpatialGraphType.RADIUS")
        indices = tree.query_ball_point(coords, r=radius)
        for i in range(num_spots):
            for neighbor in indices[i]:
                if i != neighbor:
                    edge_list.append((i, neighbor))

    elif method == SpatialGraphType.DELAUNAY:
        # Triangulation-based graph: naturally adapts to varying spot density
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            # A simplex in 2D is a triangle (3 points)
            # Create edges between all pairs in the triangle
            for i in range(3):
                for j in range(i + 1, 3):
                    edge_list.append((simplex[i], simplex[j]))
                    edge_list.append((simplex[j], simplex[i]))
        # Delaunay requires deduplication of edges shared by adjacent triangles
        edge_list = list(set(edge_list))

    else:
        raise ValueError(f"Unsupported SpatialGraphType: {method}")

    assert len(edge_list) > 0, "No edges were created in the spatial graph."

    # Convert to [2, E]
    # Ensure memory is contiguous for GCN efficiency
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    logger.debug(
        f"Spatial graph has {edge_index.size(0)} nodes and {edge_index.size(1)} edges."
    )
    return edge_index.to(device)
