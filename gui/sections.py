"""Ordered registry of the report sections shown below the headline plots.

Order mirrors ``src/analysis/report.py``. ``kind`` decides how the section is
produced:

* ``scaffold`` -- figures from ``render.ensure_scaffold_plots`` (needs the reference scaffold)
* ``disk``     -- figures from ``render.ensure_disk_plots`` (needs only the K's data dir)
* ``table``    -- a metrics table read from a JSON file in the K's data dir
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    kind: str  # "scaffold" | "disk" | "table"
    plots: tuple[str, ...] = ()  # logical plot names in render._{DISK,SCAFFOLD}_PLOTS
    table: str | None = None  # JSON filename under analysis/data/
    needs_confidence: bool = False


SECTIONS: list[Section] = [
    Section(
        "fractions", "Cell- & Spot-State Fractions", "scaffold", plots=("fractions",)
    ),
    Section(
        "leiden_merge",
        "Leiden Overclusters Merged per AIM State",
        "scaffold",
        plots=("leiden_merge",),
    ),
    Section(
        "spatial_org",
        "Spatial Organisation of Mapped Spots",
        "table",
        table="topology_metrics.json",
    ),
    Section(
        "reconstruction",
        "Reconstruction Cosine Similarity",
        "disk",
        plots=("reconstruction",),
    ),
    Section("profiles", "AIM-State Profiles", "scaffold", plots=("profiles",)),
    Section(
        "onehot",
        "Mapping Sharpness — How One-Hot",
        "disk",
        plots=("onehot_distribution", "onehot_thresholds"),
        table="onehot_summary_mapping.json",
    ),
    Section(
        "confidence",
        "Mapping Confidence — Per-Spot",
        "disk",
        plots=("confidence",),
        table="confidence_summary.json",
        needs_confidence=True,
    ),
    Section(
        "substate", "Substate Merge Coherence", "table", table="biology_metrics.json"
    ),
    Section("modularity", "Modularity", "table", table="modularity_metrics.json"),
]

SCAFFOLD_KEYS: frozenset[str] = frozenset(
    s.key for s in SECTIONS if s.kind == "scaffold"
)

# Clustering-side sections that are independent of the spot-mapping method. They
# live only in the "Single-cell reference" tab (rendered there as interactive
# Plotly), so they are excluded from every per-mapper tab and the Compare tab.
REFERENCE_SECTION_KEYS: frozenset[str] = frozenset(
    {"leiden_merge", "profiles", "substate"}
)

# Sections shown in each per-mapper tab (everything that is not reference-only).
MAPPER_SECTIONS: list[Section] = [
    s for s in SECTIONS if s.key not in REFERENCE_SECTION_KEYS
]
