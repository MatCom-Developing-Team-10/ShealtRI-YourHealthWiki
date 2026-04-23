"""Indexer module for document storage, corpus building, and persistence."""

from .document_store import (
    DocumentReadError,
    DocumentStoreError,
    DocumentWriteError,
    FileSystemDocumentStore,
)
from .index_store import IndexStore, IndexStoreError
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
    # Indexer persistence / management
    "IndexStore",
    "IndexStoreError",
]
