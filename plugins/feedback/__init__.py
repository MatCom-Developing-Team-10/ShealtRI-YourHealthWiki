"""Explicit relevance-feedback plugin for the SRI pipeline.

Public API:
    :class:`RelevanceJudgment`        — single feedback record.
    :class:`InMemoryFeedbackStore`    — non-persistent store for tests.
    :class:`JSONLFeedbackStore`       — file-backed store (one JSON per line).
    :class:`RocchioReweighter`        — pure math: q_new = αq + βμ⁺ − γμ⁻.
    :class:`RelevanceFeedbackService` — high-level orchestrator.

The plugin operates on **latent vectors** (post-LSI projection) rather than
TF-IDF, so the BaseRepository must expose ``get_embedding()`` for the
relevant / non-relevant document IDs the service is asked to apply.
"""

from .models import RelevanceJudgment
from .rocchio import RocchioReweighter
from .service import RelevanceFeedbackService
from .store import InMemoryFeedbackStore, JSONLFeedbackStore

__all__ = [
    "RelevanceJudgment",
    "InMemoryFeedbackStore",
    "JSONLFeedbackStore",
    "RocchioReweighter",
    "RelevanceFeedbackService",
]
