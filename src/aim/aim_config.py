"""Run configuration for AIM: the ``AIMConfig`` knobs, the mapper registry, and the
``build_mapper`` factory that turns a config into a ``SpotStateMapper``."""

from dataclasses import dataclass

from .mapping import GreedyMapper, LearnedMapper, SpotStateMapper

# Mapping strategies keyed by CLI name.
_MAPPERS: dict[str, type[SpotStateMapper]] = {
    GreedyMapper.name: GreedyMapper,
    LearnedMapper.name: LearnedMapper,
}

MAPPING_CHOICES = tuple(_MAPPERS)


@dataclass
class AIMConfig:
    """Per-run knobs for the AIM sweep: mapping choice, hyperparameters, and K range."""

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
        """Build the ``SpotStateMapper`` named by ``self.mapping``."""
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
