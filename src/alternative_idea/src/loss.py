import torch
import torch.nn as nn
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


def compute_state_centroids(B: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute soft cell-state centroids in embedding space.

    S[k] = weighted mean of Y over cells assigned to state k,
    i.e. S = (B / colsum(B))^T @ Y.

    Args:
        B: (C x K) cell-to-state soft assignment matrix.
        Y: (C x d) cell embeddings (precomputed, fixed).
        eps: Stability term to avoid division by zero for empty states.

    Returns:
        S: (K x d) state centroids in embedding space.
    """
    B_norm = B / (B.sum(dim=0, keepdim=True) + eps)  # (C x K), columns sum to 1
    return B_norm.T @ Y  # (K x d)


class AlternativeIdeaLoss(nn.Module):
    def __init__(
        self,
        lambda_rec_spot: float = 1.0,
        lambda_rec_gene: float = 0.0,
        lambda_state_entropy: float = 1.0,
        lambda_spot_entropy: float = 1.0,
        lambda_soft_modularity: float = 0.0,
        lambda_clust_intra: float = 0.0,
        lambda_clust_inter: float = 0.0,
        eps: float = 1e-8,
        k: int = 20,
        use_cm: bool = False,
        knn_W: Tensor | None = None,
        knn_k: Tensor | None = None,
        knn_two_m: float = 1.0,
        lambda_soft_contingency: float = 0.0,
        leiden_labels: Tensor | None = None,
        leiden_n_clusters: int = 0,
        Y_scale: float = 1.0,
    ):
        super(AlternativeIdeaLoss, self).__init__()
        self.lambda_rec_spot = lambda_rec_spot
        self.lambda_rec_gene = lambda_rec_gene
        self.lambda_state_entropy = lambda_state_entropy
        self.lambda_clust_intra = lambda_clust_intra
        self.lambda_clust_inter = lambda_clust_inter
        self.lambda_spot_entropy = lambda_spot_entropy
        self.lambda_soft_modularity = lambda_soft_modularity
        self.eps = eps
        self.lnK = torch.log(torch.tensor(k, dtype=torch.float32))
        self.use_cm = use_cm
        # KNN graph components for soft modularity (precomputed, fixed during training)
        self.knn_W = knn_W  # sparse (C x C) or None
        self.knn_k = knn_k  # dense  (C,)    or None
        self.knn_two_m = knn_two_m  # scalar float
        # Leiden over-clustering components for soft contingency (precomputed, fixed)
        self.lambda_soft_contingency = lambda_soft_contingency
        self.leiden_labels = leiden_labels  # integer (C,) or None
        self.leiden_n_clusters = leiden_n_clusters  # scalar int
        self.Y_scale = max(Y_scale, 1e-8)
        logger.debug("AlternativeIdeaLoss initialized")

    def get_rec_spot_loss(
        self, A: Tensor, B: Tensor, X_shared: Tensor, Z_shared: Tensor
    ) -> Tensor:
        """
        Eq (9): Scale-invariant reconstruction loss (Z' vs Z on shared genes).

        Args:
            A: Alignment matrix (S x C)
            X_shared: scRNA-seq reference restricted to shared genes (C x G_shared)
            Z_shared: Empirical HSO data restricted to shared genes (S x G_shared)
        """
        # --- Safety Check ---
        # Ensure that the number of genes (columns) matches
        if X_shared.shape[1] != Z_shared.shape[1]:
            raise ValueError(
                f"Gene dimension mismatch! X has {X_shared.shape[1]} genes, "
                f"but Z_marker has {Z_shared.shape[1]} genes."
            )

        # 1. Create Z_prime (The augmented spot data reconstructed from the scRNA-seq reference)
        if self.use_cm:

            C = torch.matmul(A, B)  # (S x K)

            # Compute B_normalized: Normalize B^T by the number of cells assigned to each state to prevent scale issues
            B_normalized_sum_over_types_1 = B / (
                torch.sum(B, dim=0) + self.eps
            )  # (K x C)

            M = torch.matmul(
                B_normalized_sum_over_types_1.t(), X_shared
            )  # (K x G_shared)
            # M = torch.matmul(B.t(), X_shared)  # (K x G_shared)
            Z_prime = torch.matmul(C, M)  # (S x G_shared)

        else:
            # (S x C) @ (C x G_shared) -> (S x G_shared)
            Z_prime = torch.matmul(A, X_shared)

        # 2. Compute Cosine Similarity Components
        # Dot product across the gene dimension for each spot
        dot_product = torch.sum(Z_shared * Z_prime, dim=1)  # (S,)

        # L2 Norms for each spot
        norm_Z = torch.norm(Z_shared, p=2, dim=1)  # (S,)
        norm_Z_prime = torch.norm(Z_prime, p=2, dim=1)  # (S,)

        # 3. Calculate Scale-Invariant Loss
        # We add eps to the denominator for numerical stability
        cosine_sim = dot_product / (norm_Z * norm_Z_prime + self.eps)

        # Clip to range [-1, 1] to avoid nan due to float precision before distance calc
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        # Distance = sqrt(1 - similarity)
        # loss_per_spot = torch.sqrt(1.0 - cosine_sim)  # (S,)
        # sic. mit sqrt gabs nan-probleme im traning mit den gradienten
        loss_per_spot = torch.clamp(1.0 - cosine_sim, min=0.0)

        return torch.mean(loss_per_spot)

    def get_rec_gene_loss(
        self, A: Tensor, B: Tensor, X_shared: Tensor, Z_shared: Tensor
    ) -> Tensor:
        """
        Gene-wise scale-invariant reconstruction loss (Z' vs Z on shared genes).

        Equivalent to get_rec_spot_loss but the cosine similarity is computed
        per gene (over the spots dimension) instead of per spot (over the genes
        dimension).

        Args:
            A: Alignment matrix (S x C)
            B: Cell-to-state matrix (C x K)
            X_shared: scRNA-seq reference restricted to shared genes (C x G_shared)
            Z_shared: Empirical ST data restricted to shared genes (S x G_shared)
        """
        # 1. Compute Z_prime (same as in get_rec_spot_loss)
        if self.use_cm:
            C = torch.matmul(A, B)  # (S x K)
            B_normalized = B / (
                torch.sum(B, dim=0) + self.eps
            )  # (C x K), col-normalised
            M = torch.matmul(B_normalized.t(), X_shared)  # (K x G_shared)
            Z_prime = torch.matmul(C, M)  # (S x G_shared)
        else:
            Z_prime = torch.matmul(A, X_shared)  # (S x G_shared)

        # 2. Cosine similarity gene-wise: sum over spots (dim=0)
        dot_product = torch.sum(Z_shared * Z_prime, dim=0)  # (G_shared,)
        norm_Z = torch.norm(Z_shared, p=2, dim=0)  # (G_shared,)
        norm_Z_prime = torch.norm(Z_prime, p=2, dim=0)  # (G_shared,)

        cosine_sim = dot_product / (norm_Z * norm_Z_prime + self.eps)
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        loss_per_gene = torch.clamp(1.0 - cosine_sim, min=0.0)
        return torch.mean(loss_per_gene)

    def get_state_entropy_loss(self, B: Tensor) -> Tensor:
        """
        Eq (12): Cell state entropy loss.

        Prioritizes the distribution of global cell-to-state assignments.

        Args:
            B: Cell-State assignment matrix (C x K).
        """
        # 1. Calculate p: The fraction of cells mapped to each state k
        # p is the mean of B across the cell dimension (dim=0)
        # Shape: (K,)
        p = torch.mean(B, dim=0)

        # 2. Compute Entropy: -sum(p * log(p + eps))
        # eps is added for numerical stability to avoid log(0)
        return -torch.sum(p * torch.log(p + self.eps))

    def get_spot_entropy_loss(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Eq (13): Spot state entropy loss.

        Prioritizes confident spot-to-cell-state assignments by minimizing
        the entropy of the derived spot-state matrix C.

        Args:
            A: Alignment matrix (S x C).
            B: Cell-State assignment matrix (C x K).
        """
        # 1. Compute C (S x K): Spot-to-State assignment matrix
        C = torch.matmul(A, B)

        # 2. Compute Entropy per spot
        # Formula: -sum_k(C_sk * log(C_sk + eps))
        # We perform the sum across the state dimension (dim=1)
        # Result shape: (S,)
        spot_entropies = -torch.sum(C * torch.log(C + self.eps), dim=1)

        # 3. Return the mean across all spots multiplied by lambda
        return torch.mean(spot_entropies)

    def get_clust_intra_loss(self, A: Tensor, B: Tensor, Y: Tensor) -> Tensor:
        """
        Soft within-cluster variance in embedding space, weighted by cell usage.

        L = (sum_c w_c)^{-1} sum_c w_c * sum_k B[c,k] * ||Y[c] - S[k]||^2
        where w_c = sum_s A[s,c]  (how much cell c is used across all spots).

        Args:
            A: (S x C) spot-to-cell alignment matrix.
            B: (C x K) cell-to-state soft assignment matrix.
            Y: (C x d) precomputed cell embeddings.
        """
        w = A.sum(dim=0)  # (C,) — 1^T A, total assignment weight per cell
        S = compute_state_centroids(B, Y, self.eps)  # (K x d)
        diff = Y.unsqueeze(1) - S.unsqueeze(0)  # (C x K x d)
        sq_dist = (diff**2).sum(dim=2)  # (C x K)
        return (w.unsqueeze(1) * B * sq_dist).sum() / (
            w.sum() * Y.shape[1] * self.Y_scale
        )

    def get_clust_inter_loss(self, B: Tensor, Y: Tensor) -> Tensor:
        """
        Mean pairwise squared distance between cell-state centroids, negated.

        L = -1/(K*(K-1)) * sum_{k != k'} ||S_k - S_k'||^2

        Args:
            B: (C x K) cell-to-state soft assignment matrix.
            Y: (C x d) precomputed cell embeddings.
        """
        S = compute_state_centroids(B, Y, self.eps)  # (K x d)
        K = S.shape[0]
        if K < 2:
            return torch.tensor(0.0, device=S.device)
        diff = S.unsqueeze(1) - S.unsqueeze(0)  # (K x K x d)
        sq_dist = (diff**2).sum(dim=2)  # (K x K)
        mask = 1.0 - torch.eye(K, device=S.device, dtype=S.dtype)
        return -(sq_dist * mask).sum() / (K * (K - 1))

    def get_soft_modularity_loss(self, B: Tensor) -> Tensor:
        """
        Differentiable soft modularity on the precomputed sc KNN graph.

        Replaces the hard cluster-indicator delta(c_i, c_j) with the dot product
        of soft assignment rows B[i,:] · B[j,:], making the whole expression
        differentiable through B.

        Q_soft = Tr(B^T @ (W - k*k^T / 2m) @ B) / 2m
        loss   = -Q_soft   (we minimise, so negate modularity)

        Args:
            B: Cell-to-state assignment matrix (C x K), rows are soft assignments.
        """
        # W @ B: sparse-dense matmul → (C x K)
        WB = torch.sparse.mm(self.knn_W, B)
        # Tr(B^T W B) = sum of element-wise product of B and WB
        trace_WB = torch.sum(B * WB)
        # k^T B → (K,);  Tr(B^T k k^T/2m B) = ||k^T B||^2 / 2m
        assert self.knn_k is not None
        kB = self.knn_k @ B
        trace_null = torch.dot(kB, kB) / self.knn_two_m
        Q_soft = (trace_WB - trace_null) / self.knn_two_m
        return -Q_soft

    def get_soft_contingency_loss(self, B: Tensor) -> Tensor:
        """
        Cluster-size-weighted entropy of the soft contingency matrix.

        T[l, k] = sum_{c in Leiden cluster l} B[c, k]
        For each Leiden cluster l, T[l,:] (row-normalised) should concentrate
        on one state k. We minimise the cluster-size-weighted row entropy.
        Entropy is normalised by log(K) so the value lies in [0, 1].

        Args:
            B: Cell-to-state assignment matrix (C x K).
        """
        K = B.shape[1]
        L = self.leiden_n_clusters

        # Soft contingency matrix (L x K) via scatter_add — no one-hot matrix needed
        T = torch.zeros(L, K, device=B.device)
        assert self.leiden_labels is not None
        idx = self.leiden_labels.to(B.device).unsqueeze(1).expand(-1, K)  # (C x K)
        T.scatter_add_(0, idx, B)

        # Cluster sizes: B rows sum to 1, so T[l,:].sum() = n_cells in cluster l
        sizes = T.sum(dim=1)  # (L,)
        weights = sizes / (
            sizes.sum() + self.eps
        )  # (L,) normalised cluster-size weights

        # Row-normalise T → state probability distribution per Leiden cluster
        T_norm = T / (sizes.unsqueeze(1) + self.eps)  # (L x K)

        # Per-row entropy, normalised by log(K) → range [0, 1]
        H = -(T_norm * torch.log(T_norm + self.eps)).sum(dim=1)  # (L,)
        H = H / torch.log(torch.tensor(K, dtype=torch.float32, device=B.device))

        return (weights * H).sum()

    def forward(
        self,
        A: Tensor,
        B: Tensor,
        X_shared: Tensor,
        Z_shared: Tensor,
        Y: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            A: Spot to Cell mapping (S x C)
            B: Cell to cell state mapping (C x K)
            X_shared: scRNA-seq ref on shared genes (C x g_shared)
            Z_shared: Spatial data on shared genes (S x g_shared)
            Y: Precomputed cell embeddings (C x d)
        """
        # 1. Individual terms (unweighted)
        l_rec_spot = self.get_rec_spot_loss(A, B, X_shared, Z_shared)
        l_rec_gene = self.get_rec_gene_loss(A, B, X_shared, Z_shared)
        l_state_entropy = self.get_state_entropy_loss(B)
        l_spot_entropy = self.get_spot_entropy_loss(A, B)
        l_clust_intra = self.get_clust_intra_loss(A, B, Y)
        l_clust_inter = self.get_clust_inter_loss(B, Y)
        l_soft_modularity = (
            self.get_soft_modularity_loss(B)
            if self.lambda_soft_modularity > 0.0 and self.knn_W is not None
            else B.sum()
            * 0.0  # differentiable zero — keeps grad_fn for the analysis loop
        )
        l_soft_contingency = (
            self.get_soft_contingency_loss(B)
            if self.lambda_soft_contingency > 0.0 and self.leiden_labels is not None
            else B.sum() * 0.0
        )

        # Normalize l_state_entropy and l_spot_entropy by their maximum possible values to keep them in a comparable range
        # Max entropy for both is log(K)
        l_state_entropy /= self.lnK
        l_spot_entropy /= self.lnK

        # Weighted total
        total_loss = (
            self.lambda_rec_spot * l_rec_spot
            + self.lambda_rec_gene * l_rec_gene
            + self.lambda_state_entropy * l_state_entropy
            + self.lambda_spot_entropy * l_spot_entropy
            + self.lambda_clust_intra * l_clust_intra
            + self.lambda_clust_inter * l_clust_inter
            + self.lambda_soft_modularity * l_soft_modularity
            + self.lambda_soft_contingency * l_soft_contingency
        )

        return {
            "loss": total_loss,
            "rec_spot": l_rec_spot,
            "rec_gene": l_rec_gene,
            "state_entropy": l_state_entropy,
            "spot_entropy": l_spot_entropy,
            "clust_intra": l_clust_intra,
            "clust_inter": l_clust_inter,
            "soft_modularity": l_soft_modularity,
            "soft_contingency": l_soft_contingency,
        }
