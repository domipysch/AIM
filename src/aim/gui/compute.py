"""Drive an AIM sweep for one mapper from the GUI.

Runs ``aim.run`` in-process (the same call ``aim run`` makes). The sweep only
writes machine-readable metrics per K -- no figures -- and the GUI renders every
plot on demand from them. Each mapper writes to its own run root
``<output_dir>/<mapper>/`` so several mappers coexist. The sweep runs on a
background thread; the UI polls the number of finished ``k_<kkk>/`` folders for a
progress bar.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from aim import AGGLO_TREE_METHODS, AIMConfig
from aim.cli import run_one_pair

from . import data_access

logger = logging.getLogger(__name__)

# The GUI does not expose the Leiden resolution; use the AIMConfig default so the
# sweep and the reference scaffold agree.
DEFAULT_LEIDEN_RESOLUTION = AIMConfig().leiden_resolution


class MapperRun:
    """Handle for one background mapper sweep; poll it for progress and status."""

    def __init__(
        self,
        mapper: str,
        sc_path: Path,
        st_path: Path,
        output_dir: Path,
        k_min: int | None,
        k_max: int | None,
        k_step: int,
        agglo_tree_method: str = AGGLO_TREE_METHODS[0],
        start_from_annotation: str | None = None,
    ) -> None:
        self.mapper = mapper
        self.sc_path = Path(sc_path)
        self.st_path = Path(st_path)
        self.output_dir = Path(output_dir)
        self.root = data_access.run_root(output_dir, mapper)
        self.k_min = k_min
        self.k_max = k_max
        self.k_step = k_step
        self.agglo_tree_method = agglo_tree_method
        self.start_from_annotation = start_from_annotation
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "MapperRun":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            cfg = AIMConfig(
                mapping=self.mapper,
                leiden_resolution=DEFAULT_LEIDEN_RESOLUTION,
                agglo_tree_method=self.agglo_tree_method,
                k_min=self.k_min,
                k_max=self.k_max,
                k_step=self.k_step,
                start_from_annotation=self.start_from_annotation,
            )
            # Same entry point as the CLI. The reference scaffold is cached at the
            # pair output root and shared across all mappers of this pair: build it
            # once, then reuse it.
            run_one_pair(
                self.sc_path,
                self.st_path,
                self.output_dir,
                cfg,
            )
        except Exception:  # noqa: BLE001 - surface any failure to the UI
            self.error = traceback.format_exc()
            logger.exception("Mapper sweep failed for %s", self.mapper)

    # -- progress ---------------------------------------------------------
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def n_done(self) -> int:
        """Number of K folders finished so far."""
        return len(data_access.list_ks(self.root))

    def n_expected(self) -> int | None:
        """Total K folders this sweep will produce, once the start clusters are known.

        Returns ``None`` until ``start_clustering.h5ad`` exists (the level count
        depends on the start-cluster count, which is only known then).
        """
        labels = data_access.load_start_cluster_labels(self.root)
        if labels is None:
            return None
        n_start_clusters = int(labels.max()) + 1
        k_hi = min(n_start_clusters, self.k_max) if self.k_max else n_start_clusters
        k_lo = max(1, self.k_min) if self.k_min else 1
        return len(range(k_hi, k_lo - 1, -self.k_step))
