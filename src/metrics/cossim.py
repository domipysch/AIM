"""Cosine-similarity metrics for spatial-mapping predictions.

Compares a predicted gene-expression profile (GEP) against the observed
spatial-transcriptomics (ST) counts, **per gene** and **per spot**. The same
API works for every aligner in this project (AIM, Tangram, DOT, TACCO): each
produces a genes x spots GEP as an ``AnnData``, which is all this module needs.

No normalization or log-transformation is applied — cosine similarity is
computed on the values exactly as given. Genes/spots shared between the
prediction and the ST reference are matched by name; everything else is
ignored.

Typical use::

    from metrics.cossim import compute_and_save_cossim

    result = compute_and_save_cossim(st_path, predicted_gep, metrics_folder)
    print(result.median_gene, result.median_spot)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import anndata
import numpy as np
from anndata import AnnData

from analysis.utils import to_dense

logger = logging.getLogger(__name__)

# JSON filenames written by CossimResult.save(); consumed by the analysis reports.
GENE_JSON = "cossim-per-gene{suffix}.json"
SPOT_JSON = "cossim-per-spot{suffix}.json"


# --------------------------------------------------------------------------- #
# Core numerics
# --------------------------------------------------------------------------- #
def _cosine_along_axis(A: np.ndarray, B: np.ndarray, axis: int) -> np.ndarray:
    """Row/column-wise cosine similarity of two equally-shaped 2-D arrays.

    ``axis=0`` reduces over rows -> one value per column (per gene);
    ``axis=1`` reduces over columns -> one value per row (per spot).
    Zero-norm vectors yield ``0.0``.
    """
    dot = np.sum(A * B, axis=axis)
    denom = np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.where(denom == 0.0, 0.0, dot / denom)
    return np.clip(cs, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class CossimResult:
    """Per-gene and per-spot cosine similarities for one prediction."""

    per_gene: Dict[str, float]
    per_spot: Dict[str, float]

    @property
    def median_gene(self) -> Optional[float]:
        return _median(self.per_gene)

    @property
    def median_spot(self) -> Optional[float]:
        return _median(self.per_spot)

    def save(self, output_folder: Union[str, Path], suffix: str = "") -> None:
        """Write ``cossim-per-gene{suffix}.json`` and ``cossim-per-spot{suffix}.json``.

        Each file has the shape ``{"median": float | None, "values": {name: float}}``.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        _write_json(output_folder / GENE_JSON.format(suffix=suffix), self.per_gene)
        _write_json(output_folder / SPOT_JSON.format(suffix=suffix), self.per_spot)


def _median(values: Dict[str, float]) -> Optional[float]:
    return float(np.median(list(values.values()))) if values else None


def _write_json(path: Path, values: Dict[str, float]) -> None:
    payload = {"median": _median(values), "values": values}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #
def _align(ground_truth: AnnData, prediction: AnnData) -> Tuple[AnnData, AnnData]:
    """Restrict both (spots x genes) AnnData to shared genes and spots, aligned.

    Genes and spots are matched by name and returned in the ground-truth order.
    """
    pred_genes = set(prediction.var_names)
    shared_genes = [g for g in ground_truth.var_names if g in pred_genes]
    if not shared_genes:
        raise ValueError("No shared genes between ground-truth ST and prediction.")

    if set(ground_truth.obs_names) != set(prediction.obs_names):
        raise ValueError(
            "Spot identities differ between ground-truth ST and prediction "
            f"({ground_truth.n_obs} vs {prediction.n_obs} spots)."
        )

    gt = ground_truth[:, shared_genes]
    pred = prediction[ground_truth.obs_names, shared_genes]
    return gt, pred


def compute_cossim(ground_truth: AnnData, prediction: AnnData) -> CossimResult:
    """Compute per-gene and per-spot cosine similarity.

    Args:
        ground_truth: Observed ST counts as ``AnnData`` (spots x genes).
        prediction: Predicted GEP as ``AnnData`` (genes x spots), as written by
            every aligner in this project. It is transposed internally.

    Returns:
        A :class:`CossimResult` with values keyed by gene / spot name.
    """
    gt, pred = _align(ground_truth, prediction.transpose())

    GT = to_dense(gt)  # spots x genes
    PR = to_dense(pred)  # spots x genes

    genes = list(gt.var_names)
    spots = list(gt.obs_names)

    per_gene = dict(zip(genes, _cosine_along_axis(GT, PR, axis=0).tolist()))
    per_spot = dict(zip(spots, _cosine_along_axis(GT, PR, axis=1).tolist()))
    return CossimResult(per_gene=per_gene, per_spot=per_spot)


def compute_and_save_cossim(
    st: Union[AnnData, str, Path],
    prediction: AnnData,
    output_folder: Union[str, Path, None] = None,
    suffix: str = "",
) -> CossimResult:
    """Compute cosine similarities and (optionally) save them to JSON.

    Args:
        st: Observed ST data as an ``AnnData`` or a path to an ``.h5ad`` file
            (spots x genes, raw counts).
        prediction: Predicted GEP as ``AnnData`` (genes x spots).
        output_folder: If given, write the per-gene and per-spot JSON files here.
        suffix: Appended to the JSON filenames before ``.json`` (e.g. ``"-det"``).

    Returns:
        The computed :class:`CossimResult`.
    """
    ground_truth = st if isinstance(st, AnnData) else anndata.read_h5ad(Path(st))
    result = compute_cossim(ground_truth, prediction)

    if output_folder is not None:
        result.save(output_folder, suffix=suffix)

    logger.info(
        "cossim%s — genewise median=%.4f, spotwise median=%.4f",
        suffix,
        result.median_gene if result.median_gene is not None else float("nan"),
        result.median_spot if result.median_spot is not None else float("nan"),
    )
    return result
