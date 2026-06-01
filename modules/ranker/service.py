"""Hybrid re-ranker: combines LSI similarity scores with BM25 lexical scores.

This module implements the post-retrieval re-ranking step described in the RAG
pipeline from the course lecture (Conf_9-RAG.pdf):

    Recuperación top-k → Post-proceso (re-rank, filtros) → Construcción del prompt

The HybridRanker fuses two complementary signals:
- LSI (dense): captures latent semantic similarity between query and documents.
- BM25 (sparse): captures term-level lexical overlap — exact keyword matches.

Combining both balances precision (BM25 exact match) and recall (LSI semantics),
which is the standard hybrid retrieval technique described in the lecture as
"BM25 + Embeddings, fusiones ponderadas".
"""

from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi

from core.interfaces import BaseRanker
from core.models import Query, RetrievedDocument

logger = logging.getLogger(__name__)


class HybridRanker(BaseRanker):
    """Re-ranks documents using a weighted combination of BM25 and LSI scores.

    Both score distributions are min-max normalized to [0, 1] before fusion so
    that neither signal dominates due to scale differences. The combined score
    drives the *order* of results, but the LSI similarity score on each
    document is preserved so the UI shows the absolute similarity to the
    query (not an artefact of min-max where the worst doc always lands at 0%).

    Args:
        lsi_weight: Weight for the LSI (dense) similarity score. Default 0.6.
        bm25_weight: Weight for the BM25 (sparse) lexical score. Default 0.4.

    Example:
        ranker = HybridRanker(lsi_weight=0.6, bm25_weight=0.4)
        reranked = ranker.rerank(query, retrieved_docs)
    """

    def __init__(
        self,
        lsi_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:
        if abs(lsi_weight + bm25_weight - 1.0) > 1e-6:
            raise ValueError("lsi_weight + bm25_weight must equal 1.0")
        self._lsi_weight = lsi_weight
        self._bm25_weight = bm25_weight

    # ------------------------------------------------------------------
    # BaseRanker implementation
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: Query,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Re-rank *documents* using hybrid BM25+LSI scoring.

        Args:
            query: User query; its raw text is tokenized for BM25 scoring.
            documents: Documents from the retriever in their original LSI order.

        Returns:
            Documents reordered by descending hybrid score. Each document
            preserves its original LSI similarity score so the UI surfaces
            the absolute similarity to the query rather than a normalized
            rank position.
        """
        if not documents:
            return documents

        if len(documents) == 1:
            return documents

        query_tokens = self._tokenize(query.text)
        doc_tokens = [self._tokenize(doc.document.text) for doc in documents]

        bm25_scores = self._bm25_scores(query_tokens, doc_tokens)
        lsi_scores = [doc.score for doc in documents]

        bm25_norm = self._min_max_normalize(bm25_scores)
        lsi_norm = self._min_max_normalize(lsi_scores)

        # Hybrid score drives ordering only; the LSI score on each document
        # is kept intact for display (min-max would otherwise pin the worst
        # doc to 0% regardless of its absolute similarity).
        scored = [
            (
                self._lsi_weight * lsi_norm[idx] + self._bm25_weight * bm25_norm[idx],
                doc,
            )
            for idx, doc in enumerate(documents)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        reranked = [doc for _, doc in scored]

        logger.debug(
            "HybridRanker: re-ranked %d documents for query %r",
            len(reranked),
            query.text[:60],
        )
        return reranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lightweight whitespace tokenizer (no NLP dependency for speed)."""
        return text.lower().split()

    @staticmethod
    def _bm25_scores(
        query_tokens: list[str],
        doc_tokens: list[list[str]],
    ) -> list[float]:
        """Compute BM25 scores for all documents against the query."""
        bm25 = BM25Okapi(doc_tokens)
        scores = bm25.get_scores(query_tokens)
        return scores.tolist()

    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        """Normalize *values* to [0, 1] using min-max scaling.

        If all values are identical (max == min), returns a list of 1.0 so that
        a single-signal batch still contributes its full weight.
        """
        min_v = min(values)
        max_v = max(values)
        span = max_v - min_v
        if span == 0:
            return [1.0] * len(values)
        return [(v - min_v) / span for v in values]
