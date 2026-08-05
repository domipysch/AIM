"""Run configuration for AIM: the ``AIMConfig`` knobs, the mapper registry, and the
``build_mapper`` factory that turns a config into a ``SpotStateMapper``."""

from dataclasses import dataclass

from reference_aligners.registry import REFERENCE_ALIGNERS

from .mapping import (
    NearestCentroidMapper,
    WANNMapper,
    SpotStateMapper,
    ReferenceMapper,
)

# In-process mapping strategies keyed by CLI name.
_MAPPERS: dict[str, type[SpotStateMapper]] = {
    NearestCentroidMapper.name: NearestCentroidMapper,
    WANNMapper.name: WANNMapper,
}
# External reference aligners, all served by ReferenceMapper and selected by name.
# Sourced from the aligner registry so adding one there is the only edit needed.
_REFERENCE_METHODS = tuple(REFERENCE_ALIGNERS)

MAPPING_CHOICES = tuple(_MAPPERS) + _REFERENCE_METHODS


@dataclass
class AIMConfig:
    """Per-run knobs for the AIM sweep: mapping choice, hyperparameters, and K range."""

    mapping: str = "nearest_centroid"
    leiden_resolution: float = 3.0
    # Agglomeration-tree linkage: "ward" (balanced states) or "average" (UPGMA).
    agglo_tree_method: str = "ward"
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
        return _MAPPERS[self.mapping]()
