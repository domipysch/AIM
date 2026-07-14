import torch
import torch.nn as nn
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class AIMLoss(nn.Module):
    """
    Multi-term loss for the two-level (W, G) alignment model.

    The model produces:
        G (L x L) — Leiden-subcluster -> computed-state merge
        W (S x L) — spot -> Leiden-subcluster deconvolution
        P (S x L) — spot -> computed-state = W @ G

    State gene-expression profiles are assembled from the FIXED, precomputed
    per-Leiden-cluster expression sums P_sums (L x G_shared) and sizes n (L,):

        M[k] = ( sum_l G[l,k] * P_sums[l] ) / ( sum_l G[l,k] * n_l )

    Reconstructed spot expression is  Z' = P @ M. Because P = W @ G, the spot's
    profile is built from the states its subclusters map to; merging subclusters
    that spots need to keep apart therefore hurts reconstruction, which is what
    lets the spatial data drive the merge G.

    Terms (each with a scalar lambda):
        rec_spot       — spot-level cosine reconstruction of Z' vs Z
        rec_gene       — gene-level cosine reconstruction of Z' vs Z
        state_entropy  — entropy of state-usage marginal (few states used -> merging)
        spot_entropy   — mean row-entropy of P (each spot -> one state, Shannon)
        spot_gini      — quadratic (Gini / Tsallis-2) spot sharpening on P
        merge_entropy  — size-weighted mean row-entropy of G (each subcluster -> one state, Shannon)
        merge_gini     — size-weighted quadratic (Gini / Tsallis-2) sharpening on G
        merge_coherence— penalizes co-assigning subclusters that are far apart in
                         shared-gene expression (directs *which* subclusters merge)

    The deterministic output argmaxes BOTH G and P, so making prob ≈ det requires
    both matrices to be genuinely one-hot after training. Two Gini/Tsallis-2
    sharpeners target exactly that: `spot_gini` for P (rows over states) and
    `merge_gini` for G (subcluster rows over states). Their gradient (-2 * value)
    penalises leftover second-place mass hard and uniformly, unlike Shannon
    entropy whose gradient nearly vanishes around moderately-peaked rows — which
    is why the Gini variants sharpen far more aggressively than their Shannon
    counterparts (spot_entropy / merge_entropy).
    """

    # Registered buffers (declared for type checkers; set via register_buffer).
    P_sums: Tensor
    n: Tensor
    dist: Tensor

    def __init__(
        self,
        leiden_expr_sums: Tensor,  # (L x G_shared): summed shared-gene expression per cluster
        leiden_sizes: Tensor,  # (L,): number of cells per Leiden cluster
        leiden_dist: Tensor,  # (L x L): pairwise shared-gene distance between centroids
        lambda_rec_spot: float = 0.5,
        lambda_rec_gene: float = 0.5,
        lambda_state_entropy: float = 0.1,
        lambda_spot_entropy: float = 0.0,
        lambda_spot_gini: float = 0.5,
        lambda_merge_entropy: float = 0.0,
        lambda_merge_gini: float = 1.0,
        lambda_merge_coherence: float = 0.5,
        l2_normalize_profiles: bool = False,
        eps: float = 1e-8,
    ):
        """
        Args:
            leiden_expr_sums:       Summed shared-gene expression per Leiden cluster (L x G_shared).
            leiden_sizes:           Number of cells per Leiden cluster (L,).
            leiden_dist:            Pairwise shared-gene distance between Leiden centroids (L x L).
            lambda_rec_spot:        Weight for spot-level reconstruction loss.
            lambda_rec_gene:        Weight for gene-level reconstruction loss.
            lambda_state_entropy:   Weight for state-usage entropy loss.
            lambda_spot_entropy:    Weight for the Shannon spot-state entropy loss on P.
                                    Default 0 (off) — spot_gini is the preferred P knob.
            lambda_spot_gini:       Weight for the quadratic (Gini / Tsallis-2) spot
                                    sharpening loss on P. This is the primary lever for
                                    a one-hot P.
            lambda_merge_entropy:   Weight for the Shannon Leiden-cluster merge entropy
                                    loss on G. Default 0 (off) — merge_gini is preferred.
            lambda_merge_gini:      Weight for the quadratic (Gini / Tsallis-2) merge
                                    sharpening loss on G. This is the primary lever for
                                    a one-hot G (and hence det ≈ prob).
            lambda_merge_coherence: Weight for the shared-gene merge coherence loss.
            l2_normalize_profiles:  If True, L2-normalize each state profile row of M
                                    to unit norm before the P·M reconstruction, so a
                                    state contributes in proportion to its probability
                                    weight, not its raw expression magnitude. Removes
                                    the ‖M_k‖ heterogeneity that lets a spot's residual
                                    (non-argmax) probability mass dominate the cosine
                                    direction — i.e. the source of the prob→det gap in
                                    linear (non-log) space. Default False.
            eps:                    Numerical stability constant.
        """
        super(AIMLoss, self).__init__()
        self.register_buffer("P_sums", leiden_expr_sums)  # (L x G_shared)
        self.register_buffer("n", leiden_sizes.float())  # (L,)
        self.register_buffer(
            "dist", leiden_dist
        )  # (L x L) shared-gene centroid distances
        self.lambda_rec_spot = lambda_rec_spot
        self.lambda_rec_gene = lambda_rec_gene
        self.lambda_state_entropy = lambda_state_entropy
        self.lambda_spot_entropy = lambda_spot_entropy
        self.lambda_spot_gini = lambda_spot_gini
        self.lambda_merge_entropy = lambda_merge_entropy
        self.lambda_merge_gini = lambda_merge_gini
        self.lambda_merge_coherence = lambda_merge_coherence
        self.l2_normalize_profiles = l2_normalize_profiles
        self.eps = eps
        n_leiden = int(leiden_sizes.shape[0])
        self.lnL = torch.log(torch.tensor(n_leiden, dtype=torch.float32))
        # Max Gini impurity (1 - 1/L) at a uniform row; used to normalise the
        # quadratic sharpening losses into ~[0, 1] like the entropy terms.
        self.gini_norm = 1.0 - 1.0 / n_leiden
        logger.debug("AIMLoss initialized")

    def get_state_gep(self, G: Tensor) -> Tensor:
        """
        Assemble state gene-expression profiles M (L x G_shared) from the merge
        matrix G and the fixed Leiden cluster sums/sizes.

            M[k] = (sum_l G[l,k] P_sums[l]) / (sum_l G[l,k] n_l)

        Args:
            G: Leiden-subcluster -> computed-state matrix (L x L), rows sum to 1.
        """
        weighted_sum = torch.matmul(G.t(), self.P_sums)  # (L x G_shared)
        state_sizes = torch.matmul(G.t(), self.n)  # (L,)
        M = weighted_sum / (state_sizes.unsqueeze(1) + self.eps)
        if self.l2_normalize_profiles:
            M = M / (torch.norm(M, p=2, dim=1, keepdim=True) + self.eps)
        return M

    def _cosine_rec(self, Z_shared: Tensor, Z_prime: Tensor, dim: int) -> Tensor:
        """
        Scale-invariant cosine reconstruction loss along the given dimension.

        dim=1 -> per spot (over genes); dim=0 -> per gene (over spots).
        """
        dot_product = torch.sum(Z_shared * Z_prime, dim=dim)
        norm_Z = torch.norm(Z_shared, p=2, dim=dim)
        norm_Z_prime = torch.norm(Z_prime, p=2, dim=dim)

        cosine_sim = dot_product / (norm_Z * norm_Z_prime + self.eps)
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        # 1 - cos (not sqrt(1 - cos)); sqrt caused nan gradients in training
        return torch.mean(torch.clamp(1.0 - cosine_sim, min=0.0))

    def get_rec_spot_loss(self, P: Tensor, M: Tensor, Z_shared: Tensor) -> Tensor:
        """
        Spot-level scale-invariant reconstruction loss (Z' vs Z on shared genes).

        Args:
            P: Spot -> computed-state matrix (S x L).
            M: State gene expression profiles (L x G_shared).
            Z_shared: ST data restricted to shared genes (S x G_shared).
        """
        Z_prime = torch.matmul(P, M)  # (S x G_shared)
        return self._cosine_rec(Z_shared, Z_prime, dim=1)

    def get_rec_gene_loss(self, P: Tensor, M: Tensor, Z_shared: Tensor) -> Tensor:
        """
        Gene-level scale-invariant reconstruction loss (Z' vs Z on shared genes).

        Equivalent to get_rec_spot_loss but cosine similarity is computed per
        gene (over the spot dimension) instead of per spot (over the genes
        dimension).

        Args:
            P: Spot -> computed-state matrix (S x L).
            M: State gene expression profiles (L x G_shared).
            Z_shared: ST data restricted to shared genes (S x G_shared).
        """
        Z_prime = torch.matmul(P, M)  # (S x G_shared)
        return self._cosine_rec(Z_shared, Z_prime, dim=0)

    def get_state_entropy_loss(self, G: Tensor) -> Tensor:
        """
        State-usage entropy loss. Minimize the number of used computed states.

        The state-usage marginal is the fraction of cells assigned to each state
        (via the size-weighted merge G). Low entropy -> few states used -> Leiden
        clusters are merged together.

        Args:
            G: Leiden-subcluster -> computed-state matrix (L x L).
        """
        state_sizes = torch.matmul(G.t(), self.n)  # (L,)
        p = state_sizes / (state_sizes.sum() + self.eps)  # (L,)
        return -torch.sum(p * torch.log(p + self.eps)) / self.lnL

    def get_spot_entropy_loss(self, P: Tensor) -> Tensor:
        """
        Spot-state entropy loss (Shannon). Prioritize confident (near one-hot)
        per-spot state assignments by minimizing the mean row-entropy of P.

        This is the term that, together with reconstruction, forces subclusters
        that a spot co-uses to merge: for P = W @ G to concentrate on one state,
        G must route those subclusters to the same state.

        Kept for baseline comparison; spot_gini is the preferred, stronger P
        sharpener.

        Args:
            P: Spot -> computed-state matrix (S x L).
        """
        spot_entropies = -torch.sum(P * torch.log(P + self.eps), dim=1)  # (S,)
        return torch.mean(spot_entropies) / self.lnL

    def get_spot_gini_loss(self, P: Tensor) -> Tensor:
        """
        Quadratic (Gini / Tsallis-2) spot sharpening loss.

        Mean per-spot Gini impurity  1 - sum_k P[s,k]^2, normalised by its
        maximum (1 - 1/L) so it lives in ~[0, 1] like the entropy terms.

        It is zero only when each row is one-hot. Unlike Shannon entropy, whose
        gradient nearly vanishes around moderately-peaked rows, the gradient
        here is -2 P_k, which penalises any leftover second-place mass uniformly
        and hard. (Note the identity 1 - sum_k P_k^2 = sum_k P_k (1 - P_k), an
        entrywise "push every value to 0 or 1" penalty.)

        Args:
            P: Spot -> computed-state matrix (S x L).
        """
        gini = 1.0 - torch.sum(P * P, dim=1)  # (S,)
        return torch.mean(gini) / self.gini_norm

    def get_merge_entropy_loss(self, G: Tensor) -> Tensor:
        """
        Cluster merge entropy loss (Shannon). Push each Leiden subcluster to be
        assigned to a single computed state by minimizing the cluster-size-
        weighted mean row-entropy of G.

        Kept for baseline comparison; merge_gini is the preferred, stronger G
        sharpener.

        Args:
            G: Leiden-subcluster -> computed-state matrix (L x L).
        """
        row_entropies = -torch.sum(G * torch.log(G + self.eps), dim=1)  # (L,)
        weights = self.n / (self.n.sum() + self.eps)  # (L,)
        return torch.sum(weights * row_entropies) / self.lnL

    def get_merge_gini_loss(self, G: Tensor) -> Tensor:
        """
        Quadratic (Gini / Tsallis-2) merge sharpening loss on G.

        The G analog of get_spot_gini_loss: cluster-size-weighted mean per-row
        Gini impurity  1 - sum_k G[l,k]^2, normalised by its maximum (1 - 1/L).
        Zero only when every subcluster row of G is one-hot. Because the
        deterministic prediction argmaxes G, a one-hot G is what makes the
        deterministic and probabilistic reconstructions coincide; this term is
        the primary lever for that (its -2 G_k gradient sharpens far harder than
        the Shannon merge_entropy around moderately-peaked rows).

        Args:
            G: Leiden-subcluster -> computed-state matrix (L x L).
        """
        gini = 1.0 - torch.sum(G * G, dim=1)  # (L,)
        weights = self.n / (self.n.sum() + self.eps)  # (L,)
        return torch.sum(weights * gini) / self.gini_norm

    def get_merge_coherence_loss(self, G: Tensor) -> Tensor:
        """
        Shared-gene merge coherence loss. Penalizes co-assigning subclusters that
        are far apart in shared-gene expression, so merges prefer subclusters the
        spatial (shared-gene) view cannot tell apart.

        S = G @ G^T is the soft co-assignment matrix (S[l1,l2] = probability l1
        and l2 share a state); dist is the fixed pairwise shared-gene centroid
        distance. The loss is their inner product, normalised by the number of
        subclusters. It does not drive merging itself (its own minimum is no
        co-assignment) — state_entropy drives merging, this term directs it.

        Args:
            G: Leiden-subcluster -> computed-state matrix (L x L).
        """
        S = torch.matmul(G, G.t())  # (L x L) co-assignment; diagonal is free (dist=0)
        return (self.dist * S).sum() / self.dist.shape[0]

    def forward(
        self,
        G: Tensor,
        P: Tensor,
        Z_shared: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            G: Leiden-subcluster -> computed-state mapping (L x L), rows sum to 1.
            P: Spot -> computed-state mapping (S x L) = W @ G, rows sum to 1.
            Z_shared: ST data restricted to shared genes (S x G_shared).

        Returns:
            Dict with keys:
                "loss"          — weighted total loss (scalar, use for backward).
                "rec_spot"      — unweighted spot reconstruction term.
                "rec_gene"      — unweighted gene reconstruction term.
                "state_entropy" — unweighted state-usage entropy (normalised by log L).
                "spot_entropy"  — unweighted Shannon spot entropy (normalised by log L).
                "spot_gini"     — unweighted quadratic spot sharpening term.
                "merge_entropy" — unweighted Shannon cluster merge entropy (normalised by log L).
                "merge_gini"    — unweighted quadratic merge sharpening term.
                "merge_coherence" — unweighted shared-gene merge coherence term.
        """
        M = self.get_state_gep(G)

        l_rec_spot = self.get_rec_spot_loss(P, M, Z_shared)
        l_rec_gene = self.get_rec_gene_loss(P, M, Z_shared)
        l_state_entropy = self.get_state_entropy_loss(G)
        l_spot_entropy = self.get_spot_entropy_loss(P)
        l_spot_gini = self.get_spot_gini_loss(P)
        l_merge_entropy = self.get_merge_entropy_loss(G)
        l_merge_gini = self.get_merge_gini_loss(G)
        l_merge_coherence = self.get_merge_coherence_loss(G)

        total_loss = (
            self.lambda_rec_spot * l_rec_spot
            + self.lambda_rec_gene * l_rec_gene
            + self.lambda_state_entropy * l_state_entropy
            + self.lambda_spot_entropy * l_spot_entropy
            + self.lambda_spot_gini * l_spot_gini
            + self.lambda_merge_entropy * l_merge_entropy
            + self.lambda_merge_gini * l_merge_gini
            + self.lambda_merge_coherence * l_merge_coherence
        )

        return {
            "loss": total_loss,
            "rec_spot": l_rec_spot,
            "rec_gene": l_rec_gene,
            "state_entropy": l_state_entropy,
            "spot_entropy": l_spot_entropy,
            "spot_gini": l_spot_gini,
            "merge_entropy": l_merge_entropy,
            "merge_gini": l_merge_gini,
            "merge_coherence": l_merge_coherence,
        }
