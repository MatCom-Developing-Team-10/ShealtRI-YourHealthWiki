"""Indexer module for document storage and corpus building."""

from .document_store import (
    DocumentReadError,
    DocumentStoreError,
    DocumentWriteError,
    FileSystemDocumentStore,
)
from .service import IndexerConfig, IndexerService

__all__ = [
    # Document storage
    "FileSystemDocumentStore",
    "DocumentStoreError",
    "DocumentWriteError",
    "DocumentReadError",
    # Indexer service
    "IndexerService",
    "IndexerConfig",
]
