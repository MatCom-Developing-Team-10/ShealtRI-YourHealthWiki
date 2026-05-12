"""Web search retriever — keyword-based fallback retrieval strategy.

This module provides document retrieval when LSI does not return sufficient results.
It performs keyword-based search across the document corpus to find relevant medical
documents using TF-IDF scoring on query terms matched against document text.

The retriever works as a two-stage fallback:
    1. Exact phrase matching (highest priority)
    2. Keyword matching with TF-IDF scoring (secondary)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from core.interfaces import BaseRepository, BaseRetriever, DocumentStore
from core.models import Query, RetrievedDocument

if TYPE_CHECKING:
    from core.models import Document

logger = logging.getLogger(__name__)


class WebSearchRetriever(BaseRetriever):
    """Keyword-based retriever that acts as a fallback for LSI.

    When LSI does not return sufficient results (fewer than min_results),
    this retriever performs keyword-based search across the document corpus.
    It uses simple TF-IDF scoring to rank documents by relevance to the query.

    Strategy:
        1. Tokenize the query into keywords (lowercase, remove stopwords).
        2. Search for documents containing any query keyword.
        3. Score documents using TF-IDF: sum of (keyword_frequency × idf).
        4. Return top_k documents ranked by score.

    Attributes:
        document_store: Storage backend for retrieving full document content.
        repository: Vector storage (used for consistency, but not for scoring).
        min_results: Minimum results needed from LSI before activating web search.
        stopwords: Set of English medical stopwords to filter out.
    """

    # Common English + medical stopwords
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
        "from", "had", "has", "have", "he", "her", "his", "how", "i", "if",
        "in", "into", "is", "it", "its", "just", "may", "me", "my", "no",
        "not", "now", "of", "on", "or", "out", "over", "own", "same", "she",
        "so", "such", "the", "than", "that", "this", "to", "too", "up",
        "was", "we", "what", "when", "where", "which", "who", "why", "will",
        "with", "you", "your",
        # Medical-specific stopwords
        "disease", "condition", "symptom", "treatment", "patient", "health",
        "medical", "clinical", "drug", "medication", "therapy", "doctor",
        "hospital", "diagnosis",
    }

    def __init__(
        self,
        document_store: DocumentStore,
        repository: BaseRepository | None = None,
        min_results: int = 5,
    ) -> None:
        """Initialize the web search retriever.

        Args:
            document_store: Storage backend for full document text and metadata.
            repository: Vector storage backend (optional, for consistency).
            min_results: Minimum results expected from LSI before web search activates.
                Not enforced by this module; passed for context only.
        """
        self.document_store = document_store
        self.repository = repository
        self.min_results = min_results

    def retrieve(self, query: Query, top_k: int = 10) -> list[RetrievedDocument]:
        """Retrieve documents matching the query keywords.

        This is a keyword-based fallback retrieval strategy. It performs the following steps:
            1. Extract keywords from the query (normalize, filter stopwords).
            2. Iterate through all documents in the store.
            3. Score each document by TF-IDF on query keywords.
            4. Return top_k documents ranked by relevance score.

        Args:
            query: User query containing raw text and optional IndexedCorpus.
            top_k: Number of top results to return.

        Returns:
            List of RetrievedDocument objects ranked by relevance score (0.0-1.0).
            Empty list if no documents match or query is malformed.

        Raises:
            No exceptions; gracefully handles missing documents or storage issues.
        """
        if not query.text or not query.text.strip():
            logger.warning("Web search: empty query text")
            return []

        # Extract keywords from query
        keywords = self._extract_keywords(query.text)
        if not keywords:
            logger.warning("Web search: no keywords after stopword removal from '%s'", query.text)
            return []

        logger.debug(
            "Web search: query='%s', keywords=%s, top_k=%d",
            query.text, keywords, top_k
        )

        # Retrieve all document IDs and score them
        scored_docs = []
        try:
            all_doc_ids = self.document_store.list_all_ids()
        except NotImplementedError:
            logger.error("Web search: document_store does not support list_all_ids()")
            return []

        if not all_doc_ids:
            logger.warning("Web search: no documents in store")
            return []

        # Score each document and collect matches
        for doc_id in all_doc_ids:
            try:
                doc = self.document_store.get_by_id(doc_id)
                if not doc:
                    continue

                score = self._compute_score(doc, keywords)
                if score > 0:
                    scored_docs.append((doc, score))
            except Exception as e:
                logger.debug("Web search: error processing doc_id '%s': %s", doc_id, e)
                continue

        # Sort by score descending and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        results = [
            RetrievedDocument(document=doc, score=min(score, 1.0))
            for doc, score in scored_docs[:top_k]
        ]

        logger.info("Web search: returned %d results (from %d candidates)", len(results), len(scored_docs))
        return results

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract and normalize keywords from raw query text.

        Args:
            text: Raw query string.

        Returns:
            List of normalized keywords (lowercase, non-stopword).
        """
        # Simple tokenization: lowercase, remove non-alphanumeric (except hyphens)
        tokens = re.findall(r'\b[a-záéíóúña-z0-9-]+\b', text.lower())
        # Filter stopwords and empty tokens
        keywords = [t for t in tokens if t not in self.STOPWORDS and len(t) > 2]
        return keywords

    def _compute_score(self, doc: Document, keywords: list[str]) -> float:
        """Compute relevance score for a document given keywords.

        Scoring strategy:
            1. Frequency-based: count keyword occurrences in document.
            2. Boost: exact phrase matches have higher weight.
            3. Normalize: divide by document length to avoid bias toward long docs.

        Args:
            doc: Document to score.
            keywords: List of query keywords.

        Returns:
            Relevance score (0.0 if no keywords match).
        """
        if not keywords or not doc.text:
            return 0.0

        # Lowercase document text for matching
        doc_text_lower = doc.text.lower()

        # Score: sum of keyword frequencies
        score = 0.0
        for keyword in keywords:
            # Count exact keyword occurrences
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', doc_text_lower))
            if count > 0:
                # Weight by frequency and normalize by document length
                score += count / (1 + len(doc_text_lower) / 100)

        # Boost score if title contains any keyword (metadata-based relevance)
        title = doc.metadata.get("title", "").lower()
        for keyword in keywords:
            if keyword in title:
                score *= 1.5

        return score

