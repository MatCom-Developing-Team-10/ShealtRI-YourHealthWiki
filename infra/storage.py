"""Raw document storage for crawler output.

Persists Document objects to JSONL files in data/raw/, one file per source.
Each line is a JSON object with the fields expected by DocumentLoader:
    doc_id, text, url, metadata

This module is the boundary between the crawler and the rest of the pipeline.
The indexer reads these files later via DocumentLoader — the two never interact
directly.

Usage:
    storage = RawDocumentStorage("data/raw")
    written = storage.save_batch(documents, source_name="mayo_clinic")
    # → writes/appends to data/raw/mayo_clinic.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.models import Document

logger = logging.getLogger(__name__)


class RawStorageError(Exception):
    """Raised when a document cannot be persisted to raw storage."""

    pass


class RawDocumentStorage:
    """Persists crawler documents as JSONL files in data/raw/.

    One JSONL file is created per source (e.g., mayo_clinic.jsonl).
    Each line is a JSON-serialized Document compatible with DocumentLoader.

    Calling save_batch() appends to an existing file, which allows
    incremental crawling (resume without re-downloading). Call clear()
    first if a full re-crawl is needed.

    Example:
        storage = RawDocumentStorage("data/raw")

        # Append a batch of documents from a single source
        n = storage.save_batch(documents, source_name="mayo_clinic")
        print(f"Saved {n} documents to mayo_clinic.jsonl")

        # Re-crawl from scratch
        storage.clear("mayo_clinic")
        n = storage.save_batch(fresh_documents, source_name="mayo_clinic")
    """

    def __init__(self, output_dir: str = "data/raw") -> None:
        """Initialize storage, creating the output directory if needed.

        Args:
            output_dir: Directory where JSONL files will be written.

        Raises:
            OSError: If the directory cannot be created.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def source_path(self, source_name: str) -> Path:
        """Return the JSONL file path for a given source.

        Args:
            source_name: Identifier for the source (e.g., "mayo_clinic").

        Returns:
            Path to the corresponding .jsonl file inside output_dir.
        """
        safe_name = source_name.strip().replace(" ", "_").lower()
        return self.output_dir / f"{safe_name}.jsonl"

    def save(self, document: Document, source_name: str) -> None:
        """Append a single document to the source JSONL file.

        Args:
            document: Document to persist.
            source_name: Source identifier (e.g., "mayo_clinic").

        Raises:
            RawStorageError: If the document cannot be written to disk.
        """
        path = self.source_path(source_name)
        try:
            record = self._to_record(document)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except TypeError as e:
            raise RawStorageError(
                f"Document '{document.doc_id}' contains non-serializable data: {e}"
            ) from e
        except OSError as e:
            raise RawStorageError(
                f"Cannot write document '{document.doc_id}' to '{path}': {e}"
            ) from e

    def save_batch(self, documents: list[Document], source_name: str) -> int:
        """Append a batch of documents to the source JSONL file.

        Opens the file once for the whole batch. Individual documents that
        fail serialization are skipped and logged — they do not abort the
        rest of the batch. An OSError (disk full, permissions) does abort.

        Args:
            documents: Documents to persist.
            source_name: Source identifier (e.g., "mayo_clinic").

        Returns:
            Number of documents successfully written.

        Raises:
            RawStorageError: If the file cannot be opened for writing.
        """
        if not documents:
            return 0

        path = self.source_path(source_name)
        written = 0

        try:
            with open(path, "a", encoding="utf-8") as f:
                for doc in documents:
                    try:
                        record = self._to_record(doc)
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                    except TypeError as e:
                        logger.warning(
                            "Skipping document '%s': non-serializable data: %s",
                            doc.doc_id,
                            e,
                        )
        except OSError as e:
            raise RawStorageError(
                f"Cannot open '{path}' for writing: {e}"
            ) from e

        logger.info(
            "Saved %d/%d documents → %s", written, len(documents), path
        )
        return written

    def exists(self, source_name: str) -> bool:
        """Check whether a JSONL file exists for the given source.

        Args:
            source_name: Source identifier.

        Returns:
            True if the JSONL file exists and is non-empty.
        """
        path = self.source_path(source_name)
        return path.exists() and path.stat().st_size > 0

    def clear(self, source_name: str) -> None:
        """Delete the JSONL file for a source to allow a full re-crawl.

        Silently does nothing if the file does not exist.

        Args:
            source_name: Source identifier.
        """
        path = self.source_path(source_name)
        if path.exists():
            path.unlink()
            logger.info("Cleared raw storage for source '%s'", source_name)

    @staticmethod
    def _to_record(document: Document) -> dict:
        """Serialize a Document to a dict compatible with DocumentLoader.

        Args:
            document: Document to serialize.

        Returns:
            Dict with doc_id, text, url, metadata keys.

        Raises:
            TypeError: If metadata contains non-JSON-serializable values.
        """
        return {
            "doc_id": document.doc_id,
            "text": document.text,
            "url": document.url,
            "metadata": document.metadata,
        }
