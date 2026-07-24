"""Run configuration for AIM: the ``AIMConfig`` knobs, the mapper registry, and the
``build_mapper`` factory that turns a config into a ``SpotStateMapper``."""

from dataclasses import dataclass

from .mapping import (
    NearestMapper,
    NearestScaledMapper,
    NearestEuclideanMapper,
    NearestEuclideanScaledMapper,
    LearnedMapper,
    MajorityVoteMapper,
    MajorityVoteEuclideanMapper,
    SpotStateMapper,
    ReferenceMapper,
)

# In-process mapping strategies keyed by CLI name.
_MAPPERS: dict[str, type[SpotStateMapper]] = {
    NearestMapper.name: NearestMapper,
    NearestScaledMapper.name: NearestScaledMapper,
    NearestEuclideanMapper.name: NearestEuclideanMapper,
    NearestEuclideanScaledMapper.name: NearestEuclideanScaledMapper,
    LearnedMapper.name: LearnedMapper,
    MajorityVoteMapper.name: MajorityVoteMapper,
    MajorityVoteEuclideanMapper.name: MajorityVoteEuclideanMapper,
}
# External reference aligners, all served by ReferenceMapper and selected by name.
_REFERENCE_METHODS = ("tangram", "tacco", "dot")

MAPPING_CHOICES = tuple(_MAPPERS) + _REFERENCE_METHODS


@dataclass
class AIMConfig:
    """Per-run knobs for the AIM sweep: mapping choice, hyperparameters, and K range."""

    mapping: str = "nearest"
    leiden_resolution: float = 3.0
    normalize_and_log: bool = False
    # nearest_scaled-mode only
    dispersion_shrinkage: float = 1.0
    # majority_vote-mode only
    n_neighbors: int = 10
    # learned-mode only
    epochs: int = 400
    lr: float = 0.02
    lambda_spot_gini: float = 1.0
    spot_gini_warmup_frac: float = 0.5
    # K sweep range
    k_min: int | None = None
    k_max: int | None = None
    k_step: int = 1

    def build_mapper(self) -> SpotStateMapper:
        """Build the ``SpotStateMapper`` named by ``self.mapping``."""
        if self.mapping in _REFERENCE_METHODS:
            return ReferenceMapper(reference_method=self.mapping)
        if self.mapping not in _MAPPERS:
            raise ValueError(
                f"mapping must be one of {MAPPING_CHOICES}, got {self.mapping!r}"
            )
        if self.mapping in ("nearest_scaled", "nearest_euclidean_scaled"):
            scaled_cls = {
                "nearest_scaled": NearestScaledMapper,
                "nearest_euclidean_scaled": NearestEuclideanScaledMapper,
            }[self.mapping]
            return scaled_cls(dispersion_shrinkage=self.dispersion_shrinkage)
        if self.mapping in ("majority_vote", "majority_vote_euclidean"):
            vote_cls = {
                "majority_vote": MajorityVoteMapper,
                "majority_vote_euclidean": MajorityVoteEuclideanMapper,
            }[self.mapping]
            return vote_cls(n_neighbors=self.n_neighbors)
        if self.mapping == "learned":
            return LearnedMapper(
                epochs=self.epochs,
                lr=self.lr,
                lambda_spot_gini=self.lambda_spot_gini,
                spot_gini_warmup_frac=self.spot_gini_warmup_frac,
            )
        return _MAPPERS[self.mapping]()
