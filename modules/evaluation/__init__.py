"""Offline IR evaluation module.

Computes the objective evaluation measures from the course lecture (Conf_4):
Precision, Recall, F1, NDCG, MAP and MRR over a labelled test collection.
"""

from .models import (
    EvalQuery,
    EvaluationDataset,
    EvaluationReport,
    QueryEvalResult,
)
from .service import EvaluationService

__all__ = [
    "EvalQuery",
    "EvaluationDataset",
    "EvaluationReport",
    "QueryEvalResult",
    "EvaluationService",
]
