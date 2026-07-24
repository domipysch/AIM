"""Drive an AIM sweep for one mapper from the GUI.

Runs ``aim.run`` in-process (the same call ``main.py`` makes, but with
``generate_pdf=False`` so the sweep is fast -- the GUI renders the report
sections itself). Each mapper writes to its own run root ``<output_dir>/<mapper>/``
so several mappers coexist. The sweep runs on a background thread; the UI polls
the number of finished ``k_<kkk>/`` folders for a progress bar.
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import asdict
from pathlib import Path

import yaml

from aim import AIMConfig, run as aim_run

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
    ) -> None:
        self.mapper = mapper
        self.sc_path = Path(sc_path)
        self.st_path = Path(st_path)
        self.output_dir = Path(output_dir)
        self.root = data_access.run_root(output_dir, mapper)
        self.k_min = k_min
        self.k_max = k_max
        self.k_step = k_step
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "MapperRun":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            cfg = AIMConfig(
                mapping=self.mapper,
                leiden_resolution=DEFAULT_LEIDEN_RESOLUTION,
                k_min=self.k_min,
                k_max=self.k_max,
                k_step=self.k_step,
            )
            # Mirror main.py: record the run knobs next to the outputs.
            with open(self.root / "config.yaml", "w") as f:
                yaml.safe_dump(asdict(cfg), f, sort_keys=False)

            aim_run(
                sc_path=self.sc_path,
                st_path=self.st_path,
                output_folder=self.root,
                mapper=cfg.build_mapper(),
                generate_pdf=False,
                leiden_resolution=cfg.leiden_resolution,
                k_min=self.k_min,
                k_max=self.k_max,
                k_step=self.k_step,
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
        """Total K folders this sweep will produce, once the overclustering is known.

        Returns ``None`` until ``leiden_overclustering.h5ad`` exists (the level
        count depends on the Leiden cluster count, which is only known then).
        """
        overcluster = self.root / "leiden_overclustering.h5ad"
        if not overcluster.exists():
            return None
        labels = data_access.load_leiden_labels(self.root)
        if labels is None:
            return None
        n_leiden = int(labels.max()) + 1
        k_hi = min(n_leiden, self.k_max) if self.k_max else n_leiden
        k_lo = max(1, self.k_min) if self.k_min else 1
        return len(range(k_hi, k_lo - 1, -self.k_step))
