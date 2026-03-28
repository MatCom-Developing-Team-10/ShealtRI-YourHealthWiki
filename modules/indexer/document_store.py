"""File-based document storage implementation.

Stores full document content as JSON files on disk, indexed by document ID.
This provides fast, simple document retrieval decoupled from vector search.
"""

import json
from pathlib import Path
from typing import Optional

from core.interfaces import DocumentStore
from core.models import Document


class FileSystemDocumentStore(DocumentStore):
    """Stores documents as individual JSON files on the filesystem.

    Each document is saved as {doc_id}.json in the storage directory.
    This provides:
        - Simple persistence without external dependencies
        - Fast random access by document ID
        - Easy inspection and debugging
        - Efficient for small to medium corpora (<100K documents)

    For production scale (>100K documents), consider migrating to SQLite or PostgreSQL.
    """

    def __init__(self, storage_dir: str = "data/documents") -> None:
        """Initialize the document store.

        Args:
            storage_dir: Directory path where document JSON files will be stored.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def add_documents(self, documents: list[Document]) -> None:
        """Store documents to disk as JSON files.

        Args:
            documents: List of documents to persist.
        """
        for doc in documents:
            doc_path = self._get_document_path(doc.doc_id)
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "url": doc.url,
                        "metadata": doc.metadata,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a single document by its ID.

        Args:
            doc_id: Document identifier.

        Returns:
            Document if found, None if the file doesn't exist.
        """
        doc_path = self._get_document_path(doc_id)
        if not doc_path.exists():
            return None

        with open(doc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return Document(
                doc_id=data["doc_id"],
                text=data["text"],
                url=data["url"],
                metadata=data.get("metadata", {}),
            )

    def get_by_ids(self, doc_ids: list[str]) -> list[Document]:
        """Batch retrieve multiple documents by their IDs.

        Args:
            doc_ids: List of document IDs to fetch.

        Returns:
            List of found documents (missing IDs are skipped silently).
        """
        documents = []
        for doc_id in doc_ids:
            doc = self.get_by_id(doc_id)
            if doc is not None:
                documents.append(doc)
        return documents

    def _get_document_path(self, doc_id: str) -> Path:
        """Compute the filesystem path for a document ID.

        Args:
            doc_id: Document identifier.

        Returns:
            Path object pointing to the JSON file.
        """
        # Sanitize doc_id to prevent path traversal attacks
        safe_id = doc_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_id}.json"
