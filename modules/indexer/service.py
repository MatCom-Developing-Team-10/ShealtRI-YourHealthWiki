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

        Args:
            documents: List of documents to index.

        Returns:
            IndexedCorpus with inverted_index and vocabulary.

        Raises:
            NotImplementedError: This is a stub. Implement as needed.
        """
        raise NotImplementedError(
            "IndexerService.build() must be implemented. "
            "Should return IndexedCorpus with inverted_index and vocabulary."
        )
