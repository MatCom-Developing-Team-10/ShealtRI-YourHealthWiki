"""Indexer service - builds IndexedCorpus from preprocessed documents.

This module bridges document loading and text preprocessing with the LSI retriever:
    Documents → TextProcessor → IndexerService.build() → IndexedCorpus → TfidfProcessor

It produces an IndexedCorpus with:
    - documents:        the original Document objects that survived filtering
    - processed_texts:  the preprocessed text for each document
    - inverted_index:   {term: [(doc_idx, term_frequency), ...]}
    - vocabulary:       sorted list of unique terms

Document indices in the inverted_index are contiguous and match the position of
each Document in `documents`, so they can be used directly as row indices when
constructing the TF-IDF matrix downstream.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

from core.interfaces import IndexedCorpus
from core.models import Document
from modules.text_processor import TextProcessor


logger = logging.getLogger(__name__)


@dataclass
class IndexerConfig:
    """Configuration for the indexer service.

    Attributes:
        min_document_length: Minimum number of tokens required for indexing.
            Documents shorter than this are skipped.
        min_term_frequency: Minimum total frequency (across all documents) for
            a term to be kept in the vocabulary. Helps prune noise.
        log_progress_every: Log progress every N documents.
    """

    min_document_length: int = 1
    min_term_frequency: int = 1
    log_progress_every: int = 100


class IndexerService:
    """Builds IndexedCorpus from a list of Document objects.

    The build() flow:
        1. Preprocess each document via TextProcessor (is_query=False, so tokens
           are added to the spell-checker vocabulary).
        2. Skip documents whose token count is below min_document_length.
        3. Build the inverted index using contiguous doc indices.
        4. Optionally prune low-frequency terms.
        5. Sort the vocabulary for deterministic ordering.

    The build_query() flow mirrors build() for a single query string, using
    is_query=True so the spell-checker corrects typos against the learned vocab.
    """

    def __init__(
        self,
        text_processor: TextProcessor,
        config: IndexerConfig | None = None,
    ) -> None:
        """Initialize the indexer service.

        Args:
            text_processor: TextProcessor instance for preprocessing.
            config: Indexer configuration. Uses defaults if None.
        """
        self.text_processor = text_processor
        self.config = config or IndexerConfig()

    # ------------------------------------------------------------------
    # Document indexing
    # ------------------------------------------------------------------

    def build(self, documents: list[Document]) -> IndexedCorpus:
        """Build an IndexedCorpus from raw documents.

        Args:
            documents: List of documents to index.

        Returns:
            IndexedCorpus with documents, processed_texts, inverted_index, and
            vocabulary populated. Documents shorter than min_document_length are
            silently skipped (logged at debug level).
        """
        valid_documents: list[Document] = []
        processed_texts: list[str] = []
        inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for i, doc in enumerate(documents):
            processed = self.text_processor.process(doc.text, is_query=False)
            tokens = processed.split()

            if len(tokens) < self.config.min_document_length:
                logger.debug(
                    f"Skipping doc {doc.doc_id} ({len(tokens)} tokens, "
                    f"min={self.config.min_document_length})"
                )
                continue

            doc_idx = len(valid_documents)
            for term, tf in Counter(tokens).items():
                inverted_index[term].append((doc_idx, tf))

            valid_documents.append(doc)
            processed_texts.append(processed)

            if (i + 1) % self.config.log_progress_every == 0:
                logger.info(f"Indexed {i + 1}/{len(documents)} documents")

        # Prune low-frequency terms across the whole corpus
        if self.config.min_term_frequency > 1:
            inverted_index = defaultdict(
                list,
                {
                    term: postings
                    for term, postings in inverted_index.items()
                    if sum(tf for _, tf in postings) >= self.config.min_term_frequency
                },
            )

        vocabulary = sorted(inverted_index.keys())

        logger.info(
            f"Built corpus: {len(valid_documents)} documents, "
            f"{len(vocabulary)} unique terms"
        )

        return IndexedCorpus(
            documents=valid_documents,
            processed_texts=processed_texts,
            inverted_index=dict(inverted_index),
            vocabulary=vocabulary,
        )

    # ------------------------------------------------------------------
    # Query indexing
    # ------------------------------------------------------------------

    def build_query(self, text: str) -> IndexedCorpus:
        """Build an IndexedCorpus from a raw query string.

        The query is wrapped as a single synthetic Document so the TF-IDF
        processor can consume it through the same interface as the document
        corpus. The TextProcessor runs in query mode (is_query=True) so the
        spell-checker corrects tokens against the learned document vocabulary.

        Args:
            text: Raw query string entered by the user.

        Returns:
            IndexedCorpus with a single document representing the query.
            Terms are kept regardless of min_document_length; queries are
            user input, not corpus content.
        """
        processed = self.text_processor.process(text, is_query=True)
        tokens = processed.split()

        inverted_index: dict[str, list[tuple[int, int]]] = {
            term: [(0, tf)] for term, tf in Counter(tokens).items()
        }

        query_doc = Document(doc_id="__query__", text=text, url="")

        return IndexedCorpus(
            documents=[query_doc],
            processed_texts=[processed],
            inverted_index=inverted_index,
            vocabulary=sorted(inverted_index.keys()),
        )
