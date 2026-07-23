"""Evaluation metrics for spatial-mapping predictions.

Pure computation — numbers / AnnData in, results out — shared by the AIM
post-mapping analysis (``analysis``) and the reference-aligner analysis
(``reference_aligners/mapping_analysis``). No disk-layout or method-specific
orchestration lives here.

    onehot          onehot_metrics, hard_mapping, DOMINANCE_THRESHOLDS
    cossim          CossimResult, compute_cossim, compute_and_save_cossim
    reconstruction  predict_expression, assemble_state_centroids
    biology         spatial organisation, substate coherence, modularity,
                    permutation_test
"""

from .cossim import CossimResult, compute_and_save_cossim, compute_cossim
from .onehot import DOMINANCE_THRESHOLDS, hard_mapping, onehot_metrics
from .reconstruction import assemble_state_centroids, predict_expression

__all__ = [
    "CossimResult",
    "compute_and_save_cossim",
    "compute_cossim",
    "DOMINANCE_THRESHOLDS",
    "hard_mapping",
    "onehot_metrics",
    "assemble_state_centroids",
    "predict_expression",
]
