"""Fallback retriever — chains a primary and a secondary retrieval strategy.

When the primary retriever (LSI) returns fewer than ``min_results`` documents,
the fallback retriever is activated and its results are appended. Duplicate
documents (same ``doc_id``) are deduplicated, keeping the higher score.

Typical wiring:
    primary  = LSIRetriever(...)        # fast, corpus-local
    fallback = InternetSearchRetriever(...) # slower, internet-based

    retriever = FallbackRetriever(
        primary=primary,
        fallback=fallback,
        min_results=3,   # activate internet search if LSI returns < 3 docs
    )

The ``FallbackRetriever`` itself implements ``BaseRetriever`` so it can be
dropped in anywhere a retriever is expected.

Usage:
    results = retriever.retrieve(query, top_k=10)
    # Returns up to top_k documents. Primary results appear first,
    # fallback results are appended after deduplication.
"""

from __future__ import annotations

import logging

from core.interfaces import BaseRetriever, IndexedCorpus
from core.models import Query, RetrievedDocument

logger = logging.getLogger(__name__)


class FallbackRetriever(BaseRetriever):
    """Chains two retrievers: primary first, fallback when primary is insufficient.

    Strategy:
        1. Run the primary retriever (e.g., LSI) with the full ``top_k`` budget.
        2. If the number of results is below ``min_results``, run the fallback
           (e.g., internet search) to supplement.
        3. Merge results, deduplicate by ``doc_id`` (keeping the higher score),
           sort by score descending, and return up to ``top_k`` documents.

    This keeps the primary retriever fast for the common case where the local
    corpus has enough relevant content, and only pays the cost of internet
    search when truly needed.

    Attributes:
        primary: Main retrieval strategy (typically LSI-based).
        fallback: Secondary strategy activated when primary is insufficient.
        min_results: Threshold below which the fallback is triggered.
            If ``primary`` returns fewer than this many documents, ``fallback``
            is also called and its results are merged in.
    """

    def __init__(
        self,
        primary: BaseRetriever,
        fallback: BaseRetriever,
        min_results: int = 3,
    ) -> None:
        """Initialize with a primary and fallback retriever.

        Args:
            primary: Retriever to try first (e.g., ``LSIRetriever``).
            fallback: Retriever to activate if primary returns too few results
                (e.g., ``InternetSearchRetriever``).
            min_results: Minimum number of results from ``primary`` before the
                fallback is skipped. Defaults to 3.
        """
        self.primary = primary
        self.fallback = fallback
        self.min_results = min_results

    def fit(self, corpus: IndexedCorpus) -> None:
        """Fit the primary retriever on the given corpus.

        Delegates to the primary retriever's ``fit`` method. The fallback
        retriever (e.g., ``InternetSearchRetriever``) is network-based and
        does not require fitting.

        Args:
            corpus: Preprocessed corpus from the indexer.
        """
        self.primary.fit(corpus)

    def retrieve(self, query: Query, top_k: int = 10) -> list[RetrievedDocument]:
        """Retrieve documents using primary strategy with optional fallback.

        Args:
            query: User query passed to both retrievers.
            top_k: Maximum number of results to return overall.

        Returns:
            Merged, deduplicated list of documents sorted by score (descending).
            Primary results are preferred when the same document appears in both.
        """
        # Phase 1: primary retrieval
        primary_results = self._run_primary(query, top_k)

        if len(primary_results) >= self.min_results:
            logger.debug(
                "FallbackRetriever: primary returned %d results (>= min_results=%d), "
                "skipping fallback for query='%s'",
                len(primary_results), self.min_results, query.text,
            )
            return primary_results[:top_k]

        # Phase 2: fallback retrieval
        logger.info(
            "FallbackRetriever: primary returned %d results (< min_results=%d), "
            "activating fallback for query='%s'",
            len(primary_results), self.min_results, query.text,
        )
        fallback_results = self._run_fallback(query, top_k)

        merged = self._merge(primary_results, fallback_results, top_k)
        logger.info(
            "FallbackRetriever: merged %d primary + %d fallback → %d total results",
            len(primary_results), len(fallback_results), len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_primary(self, query: Query, top_k: int) -> list[RetrievedDocument]:
        """Run the primary retriever, returning [] on any error.

        Args:
            query: User query.
            top_k: Budget for primary retriever.

        Returns:
            List of retrieved documents (may be empty).
        """
        try:
            return self.primary.retrieve(query, top_k=top_k)
        except Exception as e:
            logger.warning(
                "FallbackRetriever: primary retriever failed: %s", e
            )
            return []

    def _run_fallback(self, query: Query, top_k: int) -> list[RetrievedDocument]:
        """Run the fallback retriever, returning [] on any error.

        Args:
            query: User query.
            top_k: Budget for fallback retriever.

        Returns:
            List of retrieved documents (may be empty).
        """
        try:
            return self.fallback.retrieve(query, top_k=top_k)
        except Exception as e:
            logger.warning(
                "FallbackRetriever: fallback retriever failed: %s", e
            )
            return []

    @staticmethod
    def _merge(
        primary: list[RetrievedDocument],
        fallback: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Merge two result lists, deduplicate by doc_id, sort by score.

        When the same document appears in both lists, the entry with the
        higher score is kept.

        Args:
            primary: Results from the primary retriever.
            fallback: Results from the fallback retriever.
            top_k: Maximum number of documents to return.

        Returns:
            Merged, deduplicated list sorted by score descending.
        """
        seen: dict[str, RetrievedDocument] = {}

        for result in primary:
            doc_id = result.document.doc_id
            if doc_id not in seen or result.score > seen[doc_id].score:
                seen[doc_id] = result

        for result in fallback:
            doc_id = result.document.doc_id
            if doc_id not in seen or result.score > seen[doc_id].score:
                seen[doc_id] = result

        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        return merged[:top_k]
