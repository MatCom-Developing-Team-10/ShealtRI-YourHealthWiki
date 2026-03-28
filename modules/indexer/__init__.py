<<<<<<< HEAD
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
=======
"""Indexer module for document storage and preprocessing."""

from .document_store import FileSystemDocumentStore

__all__ = ["FileSystemDocumentStore"]
>>>>>>> 2491ed1 (feat: Enhance LSI retrieval system with new data structures and storage layers)
