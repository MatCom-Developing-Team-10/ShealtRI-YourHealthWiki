"""Document loader service for loading documents from various sources.

This module provides utilities to load Document objects from:
    - JSON files (single or directory)
    - Raw dictionaries
    - Other sources (extensible)

Usage:
    loader = DocumentLoader()
    documents = loader.load_from_directory("data/raw/")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.models import Document


logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Base exception for document loading operations."""

    pass


class DocumentLoader:
    """Loads documents from various sources into Document objects.

    Provides methods to load documents from:
        - Single JSON file containing a list of documents
        - Directory of JSON files (one document per file or lists)
        - Raw Python dictionaries

    Example:
        loader = DocumentLoader()

        # From directory
        docs = loader.load_from_directory("data/raw/", pattern="*.json")

        # From single file
        docs = loader.load_from_json("data/corpus.json")

        # From dicts
        docs = loader.load_from_list([
            {"doc_id": "1", "text": "...", "url": "..."},
        ])
    """

    def load_from_directory(
        self,
        path: str | Path,
        pattern: str = "*.json",
        recursive: bool = False,
    ) -> list[Document]:
        """Load documents from JSON files in a directory.

        Each JSON file can contain either:
            - A single document object
            - A list of document objects

        Args:
            path: Directory path to search for files.
            pattern: Glob pattern to match files (default: "*.json").
            recursive: If True, search subdirectories recursively.

        Returns:
            List of Document objects loaded from all matching files.

        Raises:
            DocumentLoaderError: If the directory doesn't exist or is not readable.
        """
        dir_path = Path(path)

        if not dir_path.exists():
            raise DocumentLoaderError(f"Directory not found: {path}")

        if not dir_path.is_dir():
            raise DocumentLoaderError(f"Path is not a directory: {path}")

        glob_method = dir_path.rglob if recursive else dir_path.glob
        files = sorted(glob_method(pattern))

        if not files:
            logger.warning(f"No files matching '{pattern}' found in {path}")
            return []

        documents: list[Document] = []
        for file_path in files:
            try:
                file_docs = self.load_from_json(file_path)
                documents.extend(file_docs)
                logger.debug(f"Loaded {len(file_docs)} documents from {file_path}")
            except DocumentLoaderError as e:
                logger.warning(f"Skipping file {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {len(files)} files in {path}")
        return documents

    def load_from_json(self, file_path: str | Path) -> list[Document]:
        """Load documents from a single JSON file.

        The JSON file can contain either:
            - A single document object: {"doc_id": "...", "text": "...", ...}
            - A list of document objects: [{"doc_id": "...", ...}, ...]

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of Document objects (even if file contains single document).

        Raises:
            DocumentLoaderError: If the file cannot be read or parsed.
        """
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoaderError(f"File not found: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DocumentLoaderError(f"Invalid JSON in {file_path}: {e}") from e
        except OSError as e:
            raise DocumentLoaderError(f"Cannot read file {file_path}: {e}") from e

        # Handle both single document and list of documents
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise DocumentLoaderError(
                f"Expected dict or list in {file_path}, got {type(data).__name__}"
            )

        return self.load_from_list(data)

    def load_from_list(self, data: list[dict]) -> list[Document]:
        """Convert raw dictionaries to Document objects.

        Args:
            data: List of dictionaries with document fields.
                Required fields: doc_id, text, url
                Optional fields: metadata

        Returns:
            List of Document objects.

        Raises:
            DocumentLoaderError: If required fields are missing.
        """
        documents: list[Document] = []

        for i, item in enumerate(data):
            try:
                doc = self._dict_to_document(item)
                documents.append(doc)
            except (KeyError, TypeError) as e:
                raise DocumentLoaderError(
                    f"Invalid document at index {i}: {e}"
                ) from e

        return documents

    def _dict_to_document(self, data: dict) -> Document:
        """Convert a single dictionary to a Document object.

        Args:
            data: Dictionary with document fields.

        Returns:
            Document object.

        Raises:
            KeyError: If required fields are missing.
            TypeError: If field types are invalid.
        """
        # Validate required fields
        required = ["doc_id", "text", "url"]
        missing = [f for f in required if f not in data]
        if missing:
            raise KeyError(f"Missing required fields: {missing}")

        return Document(
            doc_id=str(data["doc_id"]),
            text=str(data["text"]),
            url=str(data["url"]),
            metadata=data.get("metadata", {}),
        )
