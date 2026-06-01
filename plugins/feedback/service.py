"""High-level relevance-feedback service.

Orchestrates the three moving parts:

    user judgement  ──►  store
                          │
                          ▼
                 query   ──►  service.apply_to_query(...)
                                       │
                                       ▼   Rocchio
                              re-weighted query vector

Designed to be plugged into :class:`~modules.retriever.service.LSIRetriever`
via its ``feedback_service`` parameter. The retriever asks the service
"given this query text and this latent vector, what should I actually
search with?" and the service either returns the original vector
unchanged (no feedback recorded yet) or a Rocchio-reweighted version.

This service never reaches into the repository directly. Instead it
takes a *callable* ``embedding_lookup`` so callers can supply any
``doc_id -> latent vector`` resolver — typically
``BaseRepository.get_embedding``.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .models import RelevanceJudgment
from .rocchio import RocchioReweighter
from .store import FeedbackStore, InMemoryFeedbackStore

logger = logging.getLogger(__name__)

EmbeddingLookup = Callable[[str], Optional[list[float]]]


class RelevanceFeedbackService:
    """Records user judgments and applies Rocchio to upcoming queries.

    Example:
        store = InMemoryFeedbackStore()
        service = RelevanceFeedbackService(store=store)

        # During retrieval (pre-feedback) — service is a no-op.
        new_q = service.apply_to_query(
            "diabetes", original_query_vector, repo.get_embedding,
        )
        assert new_q == original_query_vector

        # User marks doc 'd1' as relevant.
        service.record("diabetes", "d1", relevant=True)

        # Next time, Rocchio kicks in.
        adjusted = service.apply_to_query(
            "diabetes", original_query_vector, repo.get_embedding,
        )
    """

    def __init__(
        self,
        store: FeedbackStore | None = None,
        rocchio: RocchioReweighter | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            store: Persistence layer for judgments. Defaults to an
                in-memory store — good for tests and short-lived sessions.
            rocchio: Reweighter strategy. Defaults to standard
                ``α=1.0, β=0.75, γ=0.15`` from Manning et al.
        """
        self.store: FeedbackStore = store if store is not None else InMemoryFeedbackStore()
        self.rocchio = rocchio if rocchio is not None else RocchioReweighter()

    # ------------------------------------------------------------------
    # Recording feedback
    # ------------------------------------------------------------------

    def record(self, query_text: str, doc_id: str, relevant: bool) -> RelevanceJudgment:
        """Record a single user judgment.

        Args:
            query_text: Query the user was running when they gave feedback.
            doc_id: Document the user judged.
            relevant: ``True`` if the user marked the doc as relevant,
                ``False`` otherwise.

        Returns:
            The persisted :class:`RelevanceJudgment` (timestamp included).
        """
        judgment = RelevanceJudgment(
            query_text=query_text, doc_id=doc_id, relevant=relevant
        )
        self.store.add(judgment)
        logger.info(
            "RelevanceFeedbackService: recorded %s judgment for query=%r doc=%s",
            "RELEVANT" if relevant else "NON-RELEVANT", query_text, doc_id,
        )
        return judgment

    def record_many(
        self,
        query_text: str,
        relevant_ids: list[str] | None = None,
        non_relevant_ids: list[str] | None = None,
    ) -> None:
        """Convenience: record a batch of judgments for the same query.

        Args:
            query_text: Query the judgments belong to.
            relevant_ids: Document IDs marked relevant.
            non_relevant_ids: Document IDs marked non-relevant.
        """
        for doc_id in relevant_ids or []:
            self.record(query_text, doc_id, relevant=True)
        for doc_id in non_relevant_ids or []:
            self.record(query_text, doc_id, relevant=False)

    # ------------------------------------------------------------------
    # Applying feedback
    # ------------------------------------------------------------------

    def has_feedback(self, query_text: str) -> bool:
        """Return True if any judgment exists for ``query_text``."""
        rel, non_rel = self.store.get_for_query(query_text)
        return bool(rel) or bool(non_rel)

    def apply_to_query(
        self,
        query_text: str,
        query_vector: list[float],
        embedding_lookup: EmbeddingLookup,
    ) -> list[float]:
        """Return ``query_vector`` with Rocchio applied if feedback exists.

        Args:
            query_text: The current query.
            query_vector: Current latent query vector (post-LSI projection).
            embedding_lookup: Callable mapping ``doc_id`` → latent vector
                (or ``None`` if unknown). Typically
                ``repository.get_embedding``.

        Returns:
            New query vector. Falls back to a copy of the input when there
            are no recorded judgments OR when none of the judged docs are
            known to the embedding lookup (rare race condition).
        """
        rel_ids, non_rel_ids = self.store.get_for_query(query_text)
        if not rel_ids and not non_rel_ids:
            return list(query_vector)

        rel_vectors = self._lookup_many(embedding_lookup, rel_ids)
        non_rel_vectors = self._lookup_many(embedding_lookup, non_rel_ids)

        if not rel_vectors and not non_rel_vectors:
            logger.warning(
                "RelevanceFeedbackService: feedback exists for %r but no embeddings "
                "could be resolved — falling back to original query vector",
                query_text,
            )
            return list(query_vector)

        adjusted = self.rocchio.reweight(query_vector, rel_vectors, non_rel_vectors)
        logger.info(
            "RelevanceFeedbackService: applied Rocchio to query=%r "
            "(rel=%d, non_rel=%d)",
            query_text, len(rel_vectors), len(non_rel_vectors),
        )
        return adjusted

    @staticmethod
    def _lookup_many(
        embedding_lookup: EmbeddingLookup, doc_ids: list[str]
    ) -> list[list[float]]:
        """Resolve doc IDs to vectors, skipping those the lookup doesn't know."""
        vectors: list[list[float]] = []
        for doc_id in doc_ids:
            vec = embedding_lookup(doc_id)
            if vec is not None:
                vectors.append(list(vec))
        return vectors
