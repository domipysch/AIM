"""Per-gene and per-spot cosine similarity between a predicted gene-expression
profile and observed ST counts. Values are compared as given, without
normalization; genes and spots are matched by name."""

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

GENE_JSON = "cossim-per-gene{suffix}.json"
SPOT_JSON = "cossim-per-spot{suffix}.json"


def _cosine_along_axis(A: np.ndarray, B: np.ndarray, axis: int) -> np.ndarray:
    """Cosine similarity of two equally-shaped 2-D arrays reduced along ``axis``.

    ``axis=0`` gives one value per column, ``axis=1`` one per row. Zero-norm
    vectors yield ``0.0``.
    """
    dot = np.sum(A * B, axis=axis)
    denom = np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.where(denom == 0.0, 0.0, dot / denom)
    return np.clip(cs, -1.0, 1.0)


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
        """Write ``cossim-per-gene{suffix}.json`` and ``cossim-per-spot{suffix}.json``
        into ``output_folder``, each shaped ``{"median": float|None, "values": {name: float}}``.
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


def _align(ground_truth: AnnData, prediction: AnnData) -> Tuple[AnnData, AnnData]:
    """Restrict both (spots x genes) AnnData to their shared genes and spots,
    matched by name and returned in ground-truth order. Raises if genes or spots
    do not overlap."""
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
    """Per-gene and per-spot cosine similarity between observed ST counts
    (``ground_truth``, spots x genes) and a predicted GEP (``prediction``,
    genes x spots; transposed internally)."""
    gt, pred = _align(ground_truth, prediction.transpose())

    GT = to_dense(gt)
    PR = to_dense(pred)

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
    """Compute cosine similarities and, if ``output_folder`` is given, write the
    per-gene and per-spot JSON files there (``suffix`` is inserted before ``.json``).

    ``st`` is an ``AnnData`` or an ``.h5ad`` path (spots x genes); ``prediction``
    is the predicted GEP (genes x spots)."""
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
