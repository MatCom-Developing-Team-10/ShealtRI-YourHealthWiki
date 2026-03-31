"""File-based document storage implementation.

Stores full document content as JSON files on disk, indexed by document ID.
This provides fast, simple document retrieval decoupled from vector search.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from core.interfaces import DocumentStore
from core.models import Document


logger = logging.getLogger(__name__)

# Maximum length for a safe filename (conservative for cross-platform compatibility)
MAX_FILENAME_LENGTH = 200

# Pattern for valid simple filenames (alphanumeric, underscore, hyphen, dot)
SAFE_FILENAME_PATTERN = re.compile(r"^[\w\-\.]+$")


class DocumentStoreError(Exception):
    """Base exception for document store operations."""

    pass


class DocumentWriteError(DocumentStoreError):
    """Raised when a document cannot be written to disk."""

    pass


class DocumentReadError(DocumentStoreError):
    """Raised when a document cannot be read from disk."""

    pass


class FileSystemDocumentStore(DocumentStore):
    """Stores documents as individual JSON files on the filesystem.

    Each document is saved as {safe_doc_id}.json in the storage directory.
    Document IDs are sanitized to prevent path traversal and ensure cross-platform
    compatibility. IDs with special characters or excessive length are hashed.

    This provides:
        - Simple persistence without external dependencies
        - Fast random access by document ID
        - Easy inspection and debugging
        - Efficient for small to medium corpora (<100K documents)
        - Security against path traversal attacks

    For production scale (>100K documents), consider migrating to SQLite or PostgreSQL.

    Raises:
        DocumentWriteError: When a document cannot be saved to disk.
        DocumentReadError: When a document cannot be read from disk.
    """

    def __init__(self, storage_dir: str = "data/documents") -> None:
        """Initialize the document store.

        Args:
            storage_dir: Directory path where document JSON files will be stored.

        Raises:
            OSError: If the directory cannot be created.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def add_documents(self, documents: list[Document]) -> None:
        """Store documents to disk as JSON files.

        Each document is serialized to JSON and saved with a sanitized filename
        based on its doc_id. Existing documents with the same ID are overwritten.

        Args:
            documents: List of documents to persist.

        Raises:
            DocumentWriteError: If any document fails to save. The error message
                includes the doc_id and underlying cause.
        """
        for doc in documents:
            doc_path = self._get_document_path(doc.doc_id)
            try:
                data = {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "url": doc.url,
                    "metadata": doc.metadata,
                }
                with open(doc_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except TypeError as e:
                # Non-serializable data in metadata
                logger.error(
                    f"Document {doc.doc_id} contains non-serializable data: {e}"
                )
                raise DocumentWriteError(
                    f"Cannot serialize document '{doc.doc_id}': {e}"
                ) from e
            except OSError as e:
                # Disk full, permission denied, etc.
                logger.error(f"Failed to write document {doc.doc_id}: {e}")
                raise DocumentWriteError(
                    f"Cannot save document '{doc.doc_id}': {e}"
                ) from e

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a single document by its ID.

        Args:
            doc_id: Document identifier.

        Returns:
            Document if found, None if the file doesn't exist.

        Raises:
            DocumentReadError: If the file exists but cannot be read or parsed.
        """
        doc_path = self._get_document_path(doc_id)
        if not doc_path.exists():
            return None

        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return Document(
                    doc_id=data["doc_id"],
                    text=data["text"],
                    url=data["url"],
                    metadata=data.get("metadata", {}),
                )
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted JSON for document {doc_id}: {e}")
            raise DocumentReadError(
                f"Cannot parse document '{doc_id}': corrupted JSON"
            ) from e
        except KeyError as e:
            logger.error(f"Missing required field in document {doc_id}: {e}")
            raise DocumentReadError(
                f"Document '{doc_id}' is missing required field: {e}"
            ) from e
        except OSError as e:
            logger.error(f"Failed to read document {doc_id}: {e}")
            raise DocumentReadError(
                f"Cannot read document '{doc_id}': {e}"
            ) from e

    def get_by_ids(self, doc_ids: list[str]) -> list[Document]:
        """Batch retrieve multiple documents by their IDs.

        Documents are returned in the same order as the input IDs.
        Missing documents are skipped silently (logged at debug level).

        Args:
            doc_ids: List of document IDs to fetch.

        Returns:
            List of found documents. Missing IDs are skipped.

        Note:
            Read errors for individual documents are logged but don't stop
            the batch operation. Only successfully read documents are returned.
        """
        documents = []
        for doc_id in doc_ids:
            try:
                doc = self.get_by_id(doc_id)
                if doc is not None:
                    documents.append(doc)
                else:
                    logger.debug(f"Document not found: {doc_id}")
            except DocumentReadError as e:
                # Log but continue with other documents
                logger.warning(f"Skipping unreadable document {doc_id}: {e}")
        return documents

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists without loading it.

        Args:
            doc_id: Document identifier.

        Returns:
            True if the document file exists, False otherwise.
        """
        return self._get_document_path(doc_id).exists()

    def delete(self, doc_id: str) -> bool:
        """Delete a document from storage.

        Args:
            doc_id: Document identifier.

        Returns:
            True if the document was deleted, False if it didn't exist.

        Raises:
            DocumentStoreError: If the file exists but cannot be deleted.
        """
        doc_path = self._get_document_path(doc_id)
        if not doc_path.exists():
            return False

        try:
            doc_path.unlink()
            return True
        except OSError as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise DocumentStoreError(
                f"Cannot delete document '{doc_id}': {e}"
            ) from e

    def _get_document_path(self, doc_id: str) -> Path:
        """Compute a safe filesystem path for a document ID.

        Sanitizes the doc_id to prevent:
            - Path traversal attacks (../, etc.)
            - Invalid filename characters on Windows (:, *, ?, etc.)
            - Excessively long filenames
            - Hidden files (starting with .)

        If the doc_id contains problematic characters or is too long,
        it's replaced with a SHA-256 hash prefix for safe storage.

        Args:
            doc_id: Document identifier (may contain any characters).

        Returns:
            Path object pointing to a safe JSON filename.
        """
        # Check if doc_id is safe to use directly as filename
        is_safe = (
            SAFE_FILENAME_PATTERN.match(doc_id)
            and len(doc_id) <= MAX_FILENAME_LENGTH
            and not doc_id.startswith(".")
            and doc_id not in (".", "..")
        )

        if is_safe:
            safe_id = doc_id
        else:
            # Use hash for problematic IDs
            hash_digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
            safe_id = hash_digest[:32]  # 32 chars is plenty for uniqueness
            logger.debug(f"Hashed doc_id '{doc_id[:50]}...' -> {safe_id}")

        return self.storage_dir / f"{safe_id}.json"
