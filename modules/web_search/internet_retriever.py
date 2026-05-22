"""Internet-based retriever — fetches real web content for a query.

This retriever calls DuckDuckGo and downloads the resulting pages to build
``RetrievedDocument`` objects on the fly. It is intended to be used as a
fallback when the local LSI retriever cannot find enough relevant documents.

Optionally, fetched documents are persisted in a ``DocumentStore`` so that
repeated identical queries hit the cache instead of the network.

Usage:
    fetcher = WebContentFetcher()
    cache = FileSystemDocumentStore("data/documents/web_search")
    retriever = InternetSearchRetriever(fetcher, document_store=cache)

    results = retriever.retrieve(Query(text="hypertension causes"), top_k=5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.interfaces import BaseRetriever, DocumentStore
from core.models import Query, RetrievedDocument
from modules.web_search.fetcher import WebContentFetcher

if TYPE_CHECKING:
    from core.models import Document

logger = logging.getLogger(__name__)


class InternetSearchRetriever(BaseRetriever):
    """Retriever that performs live internet searches via DuckDuckGo.

    For each query the retriever:
        1. Calls ``WebContentFetcher.fetch()`` to get pages from the web.
        2. Assigns rank-based relevance scores (first result scores highest).
        3. Optionally persists fetched documents in a ``DocumentStore`` cache.

    Scores are rank-based: the first result gets score 1.0 and they decay
    linearly, so the k-th result scores 1 - (k / (top_k + 1)).

    Attributes:
        fetcher: Component that handles DuckDuckGo search + page download.
        document_store: Optional cache to persist fetched documents.
            If provided, new docs are added after each successful fetch.
    """

    def __init__(
        self,
        fetcher: WebContentFetcher,
        document_store: DocumentStore | None = None,
    ) -> None:
        """Initialize the internet retriever.

        Args:
            fetcher: ``WebContentFetcher`` instance configured for HTTP timeouts
                and text extraction.
            document_store: Optional ``DocumentStore`` to persist fetched
                documents. Pass ``None`` to disable caching.
        """
        self.fetcher = fetcher
        self.document_store = document_store

    def retrieve(self, query: Query, top_k: int = 10) -> list[RetrievedDocument]:
        """Fetch pages from the internet for the given query.

        Args:
            query: User query. Only ``query.text`` is used; ``indexed_corpus``
                is not needed because this retriever bypasses LSI entirely.
            top_k: Maximum number of web pages to fetch and return.

        Returns:
            List of ``RetrievedDocument`` objects with rank-based scores,
            sorted from highest to lowest relevance. Empty list if the
            query is empty or all page fetches fail.
        """
        if not query.text or not query.text.strip():
            logger.warning("InternetSearchRetriever: empty query, returning []")
            return []

        logger.info(
            "InternetSearchRetriever: searching web for query='%s' (top_k=%d)",
            query.text, top_k,
        )

        documents = self.fetcher.fetch(query.text, max_results=top_k)
        if not documents:
            logger.warning(
                "InternetSearchRetriever: no web results for query='%s'", query.text
            )
            return []

        # Persist to cache if a store is provided
        if self.document_store is not None:
            self._cache_documents(documents)

        # Assign rank-based scores: position 0 → 1.0, position k → decays to 0
        results = []
        n = len(documents)
        for rank, doc in enumerate(documents):
            score = 1.0 - (rank / (n + 1))
            results.append(RetrievedDocument(document=doc, score=round(score, 4)))

        logger.info(
            "InternetSearchRetriever: returning %d results for query='%s'",
            len(results), query.text,
        )
        return results

    def _cache_documents(self, documents: list[Document]) -> None:
        """Persist fetched documents in the document store, skipping duplicates.

        Args:
            documents: Freshly fetched documents to cache.
        """
        new_docs = [
            doc for doc in documents
            if not self.document_store.exists(doc.doc_id)
        ]
        if new_docs:
            try:
                self.document_store.add_documents(new_docs)
                logger.debug(
                    "InternetSearchRetriever: cached %d new web documents", len(new_docs)
                )
            except Exception as e:
                logger.warning(
                    "InternetSearchRetriever: failed to cache documents: %s", e
                )
