"""Indexer service that builds IndexedCorpus from preprocessed documents.

This module bridges the document loading and text preprocessing with the LSI retriever:
    Documents → TextProcessor → IndexerService.build() → IndexedCorpus → TfidfProcessor
"""

from __future__ import annotations

from dataclasses import dataclass

from core.interfaces import IndexedCorpus
from core.models import Document
from modules.text_processor import TextProcessor


@dataclass
class IndexerConfig:
    """Configuration for the indexer service."""

    generate_vocabulary: bool = False
    min_document_length: int = 1
    log_progress_every: int = 100


class IndexerService:
    """Builds IndexedCorpus from a list of Document objects using a TextProcessor."""

    def __init__(
        self,
        text_processor: TextProcessor,
        config: IndexerConfig | None = None,
    ) -> None:
        """Initialize the indexer service."""
        raise NotImplementedError

    def build(self, documents: list[Document]) -> IndexedCorpus:
        """Build an IndexedCorpus from raw documents."""
        raise NotImplementedError
