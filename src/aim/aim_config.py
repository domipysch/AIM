"""
Run configuration for AIM.

``AIMConfig`` bundles the per-run knobs (mapping choice + hyperparameters + K
range) parsed from the CLI. The mapper registry (``_MAPPERS`` / ``MAPPING_CHOICES``)
and the ``build_mapper`` factory live here too, so the one place that knows the
set of mapping strategies is also the place that turns a config into a mapper.

Add a new strategy by subclassing ``aim.mapping.SpotStateMapper`` and registering
it in ``_MAPPERS`` below.
"""

from dataclasses import dataclass

from .mapping import GreedyMapper, LearnedMapper, SpotStateMapper

# Registry of available mapping strategies, keyed by CLI name.
_MAPPERS: dict[str, type[SpotStateMapper]] = {
    GreedyMapper.name: GreedyMapper,
    LearnedMapper.name: LearnedMapper,
}

MAPPING_CHOICES = tuple(_MAPPERS)


@dataclass
class AIMConfig:
    """Per-run knobs for the AIM agglomerative sweep (mapping choice + hyperparams)."""

    mapping: str = "greedy"
    leiden_resolution: float = 3.0
    normalize_and_log: bool = False
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
        """Instantiate the configured spot->state mapper."""
        if self.mapping not in _MAPPERS:
            raise ValueError(
                f"mapping must be one of {MAPPING_CHOICES}, got {self.mapping!r}"
            )
        if self.mapping == "learned":
            return LearnedMapper(
                epochs=self.epochs,
                lr=self.lr,
                lambda_spot_gini=self.lambda_spot_gini,
                spot_gini_warmup_frac=self.spot_gini_warmup_frac,
            )
        return _MAPPERS[self.mapping]()
