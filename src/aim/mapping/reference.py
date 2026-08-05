"""ReferenceMapper: delegate the spot->state step to an external aligner
(Tangram / TACCO / DOT), run out-of-process in that aligner's conda env."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from adata_schema import OBS_LEIDEN_ALL_GENES, OBSM_SPATIAL, UNS_SHARED_GENES
from reference_aligners.registry import REFERENCE_ALIGNERS, AlignerWorker
from .base import SpotStateMapper

logger = logging.getLogger(__name__)


class ReferenceMapper(SpotStateMapper):
    """Spot->state mapper that delegates to Tangram / TACCO / DOT.

    The external aligners map ST spots onto categorical cell types, so for each K
    we label the reference cells by their AIM state and hand that column to the
    aligner as the cell-type key -- the aligner's output columns are then the K
    states. Because the aligners live in their own conda environments (and DOT's
    core is in R), they run out-of-process via ``conda run`` against a shared-gene
    sc/st pair materialised once by ``prepare``.
    """

    def __init__(self, reference_method: str = "tangram") -> None:
        if reference_method not in REFERENCE_ALIGNERS:
            raise ValueError(
                f"reference_method must be one of {tuple(REFERENCE_ALIGNERS)}, "
                f"got {reference_method!r}"
            )
        self.reference_method = reference_method
        # Output-subfolder name = the chosen aligner (``tangram``/``tacco``/``dot``).
        # The sweep writes results to ``<root>/<mapper.name>/`` while ``main.py``
        # writes ``config.yaml`` to ``<root>/<config.mapping>/``; naming the mapper
        # after the aligner keeps the two in the same folder and stops the three
        # aligners from colliding in a single ``reference/`` directory.
        self.name = reference_method
        self._prepared = False
        self._worker: AlignerWorker | None = None

    @staticmethod
    def _state_key(k: int) -> str:
        """Obs-column / cell-type-key name holding the K-state labels for level k."""
        return f"state_k{k:03d}"

    def prepare(self, adata_sc, adata_st, labels_by_k) -> None:
        """Materialise the shared-gene sc/st inputs once for the whole sweep.

        The sc file carries one categorical obs column per swept K
        (``state_k{kkk}``) holding each cell's AIM state at that K, so every later
        per-K aligner run just points ``--cell_type_key`` at the right column and
        no large file is rewritten inside the loop.
        """
        super().prepare(adata_sc, adata_st, labels_by_k)
        shared = list(adata_sc.uns[UNS_SHARED_GENES])
        # Kept alive on the instance so it survives the whole sweep, then cleaned
        # up when this mapper is garbage-collected (a fresh mapper per pair).
        self._tmpdir = tempfile.TemporaryDirectory(prefix="aim_reference_")
        self._workdir = Path(self._tmpdir.name)
        self._sc_path = self._workdir / "sc_ref.h5ad"
        self._st_path = self._workdir / "st_ref.h5ad"

        # Per-cell AIM state at each K = that K's subcluster->state cut indexed by
        # every cell's Leiden over-cluster label.
        leiden = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
        sc_obs = pd.DataFrame(index=adata_sc.obs_names)
        for k, labels_k in labels_by_k.items():
            cell_states = np.asarray(labels_k)[leiden]
            sc_obs[self._state_key(k)] = pd.Categorical(cell_states.astype(str))

        ad.AnnData(
            X=adata_sc[:, shared].X.copy(),
            obs=sc_obs,
            var=pd.DataFrame(index=shared),
        ).write_h5ad(self._sc_path)

        # Carry spatial coordinates through: spatially-aware aligners (DOT)
        # need them; the others simply ignore the extra obsm entry.
        st_obsm = {}
        if OBSM_SPATIAL in adata_st.obsm:
            st_obsm[OBSM_SPATIAL] = np.asarray(adata_st.obsm[OBSM_SPATIAL])
        ad.AnnData(
            X=adata_st[:, shared].X.copy(),
            obs=pd.DataFrame(index=adata_st.obs_names),
            var=pd.DataFrame(index=shared),
            obsm=st_obsm or None,
        ).write_h5ad(self._st_path)

        self._st_obs_names = [str(s) for s in adata_st.obs_names]

        # One long-lived aligner process for the whole K-loop: it loads the
        # shared-gene pair once and maps every K, instead of a fresh `conda run`
        # (env activation + heavy imports + h5ad reload) per K.
        self._worker = AlignerWorker(
            self.reference_method, self._sc_path, self._st_path
        )
        self._worker.start()

        self._prepared = True
        logger.info(
            "ReferenceMapper[%s] prepared shared-gene inputs (%d genes, %d K-levels) at %s",
            self.reference_method,
            len(shared),
            len(labels_by_k),
            self._workdir,
        )

    def map(self, leiden_to_state, k) -> tuple[torch.Tensor, None]:
        """Delegate to the external aligner and return ``(P, None)`` — the
        reference aligners do not expose a per-spot confidence."""
        if not self._prepared:
            raise RuntimeError(
                "ReferenceMapper.prepare(...) must run before map(); the sweep "
                "calls it once before the K-loop."
            )
        k = int(k)
        n_spots = int(self.adata_st.n_obs)
        if k < 2:
            # A single state is trivial (and degenerate for the aligners).
            return torch.ones((n_spots, 1), dtype=torch.float32), None

        out_dir = self._workdir / self._state_key(k)
        logger.info("ReferenceMapper[%s] K=%d", self.reference_method, k)
        assert self._worker is not None  # set in prepare()
        out_path = self._worker.run_job(self._state_key(k), out_dir)
        return self._read_mapping(out_path, k), None

    def close(self) -> None:
        """Stop the persistent aligner worker (called by the sweep after the
        K-loop; also invoked on garbage collection as a backstop)."""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup during GC
            pass

    def _read_mapping(self, path: Path, k: int) -> torch.Tensor:
        """Load the aligner's S x (states-present) mapping_prob.h5ad and reindex it
        into a dense (S x K) matrix aligned to the ST spot order and states 0..K-1
        (states with no assigned mass come back as zero columns).

        The aligner's output is returned as-is (not re-normalised): the one-hotness
        metrics row-normalise internally, argmax is scale-invariant, and this keeps
        each aligner's native output verbatim."""
        if not path.exists():
            raise RuntimeError(f"{self.reference_method} produced no mapping at {path}")
        mp = ad.read_h5ad(path)
        X = mp.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        frame = pd.DataFrame(
            X, index=mp.obs_names.astype(str), columns=mp.var_names.astype(str)
        )
        frame = frame.reindex(
            index=self._st_obs_names,
            columns=[str(i) for i in range(k)],
            fill_value=0.0,
        )
        return torch.tensor(frame.to_numpy(dtype=np.float32), dtype=torch.float32)
