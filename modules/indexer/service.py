"""Indexer service stub - builds IndexedCorpus from preprocessed documents.

This module bridges document loading and text preprocessing with the LSI retriever:
    Documents → TextProcessor → IndexerService.build() → IndexedCorpus → TfidfProcessor

Note: The implementation is left to the user. The IndexerService should:
    1. Preprocess documents using TextProcessor
    2. Build inverted index: term → [(doc_index, term_frequency), ...]
    3. Construct vocabulary (sorted list of unique terms)
    4. Return IndexedCorpus
"""

from __future__ import annotations

from dataclasses import dataclass

from core.interfaces import IndexedCorpus
from core.models import Document
from modules.text_processor import TextProcessor


@dataclass
class IndexerConfig:
    """Configuration for the indexer service.

    Attributes:
        min_document_length: Minimum number of tokens required for indexing.
        min_term_frequency: Minimum total frequency for a term to be included.
        log_progress_every: Log progress every N documents.
    """

    min_document_length: int = 1
    min_term_frequency: int = 1
    log_progress_every: int = 100


class IndexerService:
    """Builds IndexedCorpus from a list of Document objects.

    The implementation should:
        1. Use TextProcessor to preprocess each document
        2. Build inverted index: term → [(doc_index, freq), ...]
        3. Create sorted vocabulary
        4. Return IndexedCorpus with all fields populated
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

    def build(self, documents: list[Document]) -> IndexedCorpus:
        """Build an IndexedCorpus from raw documents.

        Steps:
            1. Preprocess each document using TextProcessor (is_query=False).
            2. Build inverted index: term → [(doc_index, term_frequency), ...]
            3. Create sorted vocabulary list of unique terms.
            4. Return IndexedCorpus with all fields.

        Args:
            documents: List of documents to index.

        Returns:
            IndexedCorpus with inverted_index and vocabulary.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot index empty document list.")

        processed_texts: list[str] = []
        inverted_index: dict[str, list[tuple[int, int]]] = {}

        # Step 1: Preprocess all documents
        for doc_idx, doc in enumerate(documents):
            if doc_idx % self.config.log_progress_every == 0:
                print(f"Indexing document {doc_idx + 1}/{len(documents)}...")

            # Preprocess the document using TextProcessor (is_query=False)
            processed = self.text_processor.process(doc.text, is_query=False)

            # Validate minimum document length
            token_count = len(processed.split())
            if token_count < self.config.min_document_length:
                print(f"  Warning: Document '{doc.doc_id}' has {token_count} tokens, "
                      f"below minimum {self.config.min_document_length}. Skipping.")
                continue

            processed_texts.append(processed)

            # Step 2: Build inverted index from token frequencies
            # Count token occurrences in this document
            token_freq: dict[str, int] = {}
            for token in processed.split():
                token_freq[token] = token_freq.get(token, 0) + 1

            # Add to inverted index
            for token, freq in token_freq.items():
                if token not in inverted_index:
                    inverted_index[token] = []
                inverted_index[token].append((doc_idx, freq))

        # Step 3: Extract and sort vocabulary
        # Filter by minimum term frequency
        vocabulary = sorted([
            term for term, postings in inverted_index.items()
            if len(postings) >= self.config.min_term_frequency
        ])

        # Re-filter inverted index to match vocabulary
        filtered_inverted_index = {
            term: inverted_index[term]
            for term in vocabulary
        }

        return IndexedCorpus(
            documents=documents,
            processed_texts=processed_texts,
            inverted_index=filtered_inverted_index,
            vocabulary=vocabulary,
        )
