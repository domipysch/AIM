"""Run configuration for AIM: the ``AIMConfig`` knobs, the mapper registry, and the
``build_mapper`` factory that turns a config into a ``SpotStateMapper``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aim.reference_aligners.registry import REFERENCE_ALIGNERS

if TYPE_CHECKING:
    from .mapping import SpotStateMapper

# In-process mapping strategy names. Kept as plain strings (not derived from the
# mapper classes) so ``MAPPING_CHOICES`` — used to build the CLI parser — needs no
# import of the mapper modules. ``build_mapper`` below asserts these stay in sync
# with the classes' own ``.name``.
_INPROCESS_METHODS = ("nearest_centroid", "wann")
# External reference aligners, all served by ReferenceMapper and selected by name.
# Sourced from the aligner registry so adding one there is the only edit needed.
_REFERENCE_METHODS = tuple(REFERENCE_ALIGNERS)

MAPPING_CHOICES = _INPROCESS_METHODS + _REFERENCE_METHODS


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

    def build_mapper(self) -> "SpotStateMapper":
        """Build the ``SpotStateMapper`` named by ``self.mapping``.

        The mapper modules are imported here, lazily, so merely reading
        ``AIMConfig``/``MAPPING_CHOICES`` does not import them.
        """
        from .mapping import (
            NearestCentroidMapper,
            ReferenceMapper,
            WANNMapper,
        )

        inprocess: dict[str, type[SpotStateMapper]] = {
            NearestCentroidMapper.name: NearestCentroidMapper,
            WANNMapper.name: WANNMapper,
        }
        assert tuple(inprocess) == _INPROCESS_METHODS, (
            "MAPPING_CHOICES out of sync with mapper class names: "
            f"{tuple(inprocess)} != {_INPROCESS_METHODS}"
        )

        if self.mapping in _REFERENCE_METHODS:
            return ReferenceMapper(reference_method=self.mapping)
        if self.mapping not in inprocess:
            raise ValueError(
                f"mapping must be one of {MAPPING_CHOICES}, got {self.mapping!r}"
            )
        return inprocess[self.mapping]()
