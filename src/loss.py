import torch
import torch.nn as nn
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class AIMLoss(nn.Module):
    """
    Multi-term loss for the alignment model.

    Combines up to seven terms, each controlled by a scalar weight (lambda).
    Setting a weight to 0 disables that term at no extra compute cost (except
    soft_contingency, which is skipped entirely when its lambda is 0).

    Active terms and their roles:
        rec_spot          — spot-level cosine reconstruction
        rec_gene          — gene-level cosine reconstruction
        state_entropy     — minimize number of used cell-states
        spot_entropy      — minimize uncertainty of per-spot state assignments
        soft_contingency  — align model states with Leiden over-clustering
    """

    def __init__(
        self,
        lambda_rec_spot: float = 1.0,
        lambda_rec_gene: float = 0.0,
        lambda_state_entropy: float = 1.0,
        lambda_spot_entropy: float = 1.0,
        eps: float = 1e-8,
        n_states: int = 1,
        lambda_soft_contingency: float = 0.0,
        leiden_labels: Tensor | None = None,
    ):
        """
        Args:
            lambda_rec_spot:        Weight for spot-level reconstruction loss.
            lambda_rec_gene:        Weight for gene-level reconstruction loss.
            lambda_state_entropy:   Weight for cell-state entropy loss.
            lambda_spot_entropy:    Weight for spot-state entropy loss.
            eps:                    Numerical stability constant.
            n_states:               Number of cell states (= #Leiden clusters); used to normalize entropy terms.
            lambda_soft_contingency: Weight for soft contingency loss (0 disables it).
            leiden_labels:          Integer tensor of shape (C,) with Leiden cluster IDs.
                                    Required when lambda_soft_contingency > 0.
        """
        super(AIMLoss, self).__init__()
        self.lambda_rec_spot = lambda_rec_spot
        self.lambda_rec_gene = lambda_rec_gene
        self.lambda_state_entropy = lambda_state_entropy
        self.lambda_spot_entropy = lambda_spot_entropy
        self.eps = eps
        self.lnK = torch.log(torch.tensor(n_states, dtype=torch.float32))
        # Leiden over-clustering components for soft contingency (precomputed, fixed)
        self.lambda_soft_contingency = lambda_soft_contingency
        self.leiden_labels = leiden_labels  # integer (C,) or None
        logger.debug("AIMLoss initialized")

    def get_rec_spot_loss(
        self, A: Tensor, B: Tensor, X_shared: Tensor, Z_shared: Tensor
    ) -> Tensor:
        """
        Scale-invariant reconstruction loss (Z' vs Z on shared genes).

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
        C = torch.matmul(A, B)  # (S x K)

        # Compute B_normalized: Normalize B^T by the number of cells assigned to each state to prevent scale issues
        B_normalized_sum_over_types_1 = B / (torch.sum(B, dim=0) + self.eps)  # (K x C)

        M = torch.matmul(B_normalized_sum_over_types_1.t(), X_shared)  # (K x G_shared)
        # M = torch.matmul(B.t(), X_shared)  # (K x G_shared)
        Z_prime = torch.matmul(C, M)  # (S x G_shared)

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
        C = torch.matmul(A, B)  # (S x K)
        B_normalized = B / (torch.sum(B, dim=0) + self.eps)  # (C x K), col-normalised
        M = torch.matmul(B_normalized.t(), X_shared)  # (K x G_shared)
        Z_prime = torch.matmul(C, M)  # (S x G_shared)

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
        Cell state entropy loss.
        Minimize number of used cell-states.

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
        Spot state entropy loss.

        Prioritize confident spot-to-cell-state assignments by minimizing
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

    def get_soft_contingency_loss(self, B: Tensor) -> Tensor:
        """
        Cluster-size-weighted entropy of the soft contingency matrix.

        T[l, k] = sum_{c in Leiden cluster l} B[c, k]
        For each Leiden cluster l, T[l,:] (row-normalized) should concentrate
        on one state k. We minimie the cluster-size-weighted row entropy.
        Entropy is normalized by log(K) so the value lies in [0, 1].

        Args:
            B: Cell-to-state assignment matrix (C x K).
        """
        # n_states == n_leiden_clusters by construction → T is square
        K = B.shape[1]

        # Soft contingency matrix (K x K) via scatter_add — no one-hot matrix needed
        T = torch.zeros(K, K, device=B.device)
        assert self.leiden_labels is not None
        idx = self.leiden_labels.to(B.device).unsqueeze(1).expand(-1, K)  # (C x K)
        T.scatter_add_(0, idx, B)

        # Cluster sizes: B rows sum to 1, so T[l,:].sum() = n_cells in cluster l
        sizes = T.sum(dim=1)  # (K,)
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
    ) -> dict[str, Tensor]:
        """
        Args:
            A: Spot-to-cell mapping (S × C), rows sum to 1.
            B: Cell-to-state mapping (C × K), rows sum to 1.
            X_shared: scRNA-seq reference restricted to shared genes (C × G_shared).
            Z_shared: ST data restricted to shared genes (S × G_shared).

        Returns:
            Dict with keys:
                "loss"             — weighted total loss (scalar, use for backward).
                "rec_spot"         — unweighted spot reconstruction term.
                "rec_gene"         — unweighted gene reconstruction term.
                "state_entropy"    — unweighted state entropy term (normalised by log K).
                "spot_entropy"     — unweighted spot entropy term (normalised by log K).
                "soft_contingency" — unweighted soft contingency term (0 if disabled).
        """
        # 1. Individual terms (unweighted)
        l_rec_spot = self.get_rec_spot_loss(A, B, X_shared, Z_shared)
        l_rec_gene = self.get_rec_gene_loss(A, B, X_shared, Z_shared)
        l_state_entropy = self.get_state_entropy_loss(B)
        l_spot_entropy = self.get_spot_entropy_loss(A, B)
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
            + self.lambda_soft_contingency * l_soft_contingency
        )

        return {
            "loss": total_loss,
            "rec_spot": l_rec_spot,
            "rec_gene": l_rec_gene,
            "state_entropy": l_state_entropy,
            "spot_entropy": l_spot_entropy,
            "soft_contingency": l_soft_contingency,
        }
