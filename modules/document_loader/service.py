"""Document loader service using LangChain for multi-format support.

This module provides utilities to load Document objects from:
    - JSON files (single or directory)
    - HTML files (from crawler or web sources)
    - PDF files (medical articles, papers)
    - Plain text files
    - CSV/TSV files (medical datasets)
    - Markdown files (documentation)

Leverages LangChain's DocumentLoader ecosystem for format handling.

Usage:
    loader = DocumentLoader()

    # Auto-detect format from extension
    documents = loader.load_from_directory("data/raw/")

    # Explicit format
    documents = loader.load_from_directory("data/pdfs/", format="pdf")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Type

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain.schema import Document as LCDocument

from core.models import Document


logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Base exception for document loading operations."""

    pass


class DocumentLoader:
    """Loads documents from various formats using LangChain loaders.

    Supports:
        - JSON: Structured medical data
        - HTML: Crawler output, web pages
        - PDF: Medical articles, papers
        - TXT: Plain text documents
        - CSV/TSV: Medical datasets
        - Markdown: Documentation

    Example:
        loader = DocumentLoader()

        # Load all files from directory (auto-detect format)
        docs = loader.load_from_directory("data/raw/")

        # Load specific format
        docs = loader.load_from_directory("data/pdfs/", format="pdf")

        # Load single file
        docs = loader.load_from_file("data/article.pdf")

        # Load from JSON (backward compatible)
        docs = loader.load_from_json("data/corpus.json")
    """

    # Mapping of file extensions to loader classes
    LOADER_MAP: dict[str, Type[BaseLoader]] = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".csv": CSVLoader,
        ".tsv": CSVLoader,
    }

    def __init__(self) -> None:
        """Initialize the document loader."""
        # Try to import optional loaders
        self._register_optional_loaders()

    def _register_optional_loaders(self) -> None:
        """Register loaders that require optional dependencies."""
        # HTML loader (requires unstructured)
        try:
            from langchain_community.document_loaders import UnstructuredHTMLLoader

            self.LOADER_MAP[".html"] = UnstructuredHTMLLoader
            self.LOADER_MAP[".htm"] = UnstructuredHTMLLoader
        except ImportError:
            logger.warning(
                "UnstructuredHTMLLoader not available. Install with: pip install unstructured"
            )

        # PDF loader (requires pypdf)
        try:
            from langchain_community.document_loaders import PyPDFLoader

            self.LOADER_MAP[".pdf"] = PyPDFLoader
        except ImportError:
            logger.warning("PyPDFLoader not available. Install with: pip install pypdf")

    def load_from_directory(
        self,
        path: str | Path,
        pattern: str = "**/*",
        format: str | None = None,
        recursive: bool = True,
    ) -> list[Document]:
        """Load documents from all supported files in a directory.

        Args:
            path: Directory path to search for files.
            pattern: Glob pattern to match files (default: "**/*" for all files).
            format: Optional format hint ("json", "pdf", "html", "txt", "csv").
                If None, auto-detects from file extensions.
            recursive: If True, search subdirectories recursively (default: True).

        Returns:
            List of Document objects loaded from all matching files.

        Raises:
            DocumentLoaderError: If the directory doesn't exist or no files found.

        Example:
            # Load all supported files
            docs = loader.load_from_directory("data/")

            # Load only PDFs
            docs = loader.load_from_directory("data/", pattern="**/*.pdf")

            # Or use format hint
            docs = loader.load_from_directory("data/pdfs/", format="pdf")
        """
        dir_path = Path(path)

        if not dir_path.exists():
            raise DocumentLoaderError(f"Directory not found: {path}")

        if not dir_path.is_dir():
            raise DocumentLoaderError(f"Path is not a directory: {path}")

        # If format is specified, use pattern matching
        if format:
            return self._load_directory_by_format(dir_path, format, pattern, recursive)

        # Otherwise, load all supported files
        return self._load_directory_auto(dir_path, pattern, recursive)

    def _load_directory_by_format(
        self,
        dir_path: Path,
        format: str,
        pattern: str,
        recursive: bool,
    ) -> list[Document]:
        """Load directory with specific format."""
        if format == "json":
            return self._load_json_directory(dir_path, pattern, recursive)

        # Get loader class for format
        ext = f".{format}"
        loader_cls = self.LOADER_MAP.get(ext)

        if not loader_cls:
            raise DocumentLoaderError(
                f"Unsupported format: {format}. "
                f"Supported: {list(self.LOADER_MAP.keys())}"
            )

        # Use DirectoryLoader
        loader = DirectoryLoader(
            str(dir_path),
            glob=pattern if pattern != "**/*" else f"**/*.{format}",
            loader_cls=loader_cls,
            recursive=recursive,
            show_progress=True,
        )

        lc_docs = loader.load()
        return [self._convert_document(doc) for doc in lc_docs]

    def _load_directory_auto(
        self,
        dir_path: Path,
        pattern: str,
        recursive: bool,
    ) -> list[Document]:
        """Load directory auto-detecting formats."""
        documents: list[Document] = []

        glob_method = dir_path.rglob if recursive else dir_path.glob
        files = sorted(glob_method(pattern))

        if not files:
            logger.warning(f"No files matching '{pattern}' found in {dir_path}")
            return []

        for file_path in files:
            if not file_path.is_file():
                continue

            try:
                file_docs = self.load_from_file(file_path)
                documents.extend(file_docs)
                logger.debug(f"Loaded {len(file_docs)} documents from {file_path}")
            except DocumentLoaderError as e:
                logger.warning(f"Skipping file {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {dir_path}")
        return documents

    def load_from_file(self, file_path: str | Path) -> list[Document]:
        """Load documents from a single file (auto-detect format).

        Args:
            file_path: Path to the file.

        Returns:
            List of Document objects (may be multiple for PDFs, CSVs).

        Raises:
            DocumentLoaderError: If file not found or format unsupported.

        Example:
            docs = loader.load_from_file("data/article.pdf")
            docs = loader.load_from_file("data/corpus.json")
        """
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoaderError(f"File not found: {file_path}")

        # Special handling for JSON (backward compatibility)
        if path.suffix == ".json":
            return self.load_from_json(path)

        # Get loader for extension
        loader_cls = self.LOADER_MAP.get(path.suffix)

        if not loader_cls:
            raise DocumentLoaderError(
                f"Unsupported file extension: {path.suffix}. "
                f"Supported: {list(self.LOADER_MAP.keys()) + ['.json']}"
            )

        # Load using LangChain loader
        loader = loader_cls(str(path))
        lc_docs = loader.load()

        return [self._convert_document(doc) for doc in lc_docs]

    def load_from_json(self, file_path: str | Path) -> list[Document]:
        """Load documents from a JSON file (backward compatible).

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

    def _load_json_directory(
        self,
        dir_path: Path,
        pattern: str,
        recursive: bool,
    ) -> list[Document]:
        """Load JSON files from directory."""
        documents: list[Document] = []

        glob_method = dir_path.rglob if recursive else dir_path.glob
        json_pattern = pattern if "*.json" in pattern else "**/*.json"
        files = sorted(glob_method(json_pattern))

        if not files:
            logger.warning(f"No JSON files found in {dir_path}")
            return []

        for file_path in files:
            try:
                file_docs = self.load_from_json(file_path)
                documents.extend(file_docs)
                logger.debug(f"Loaded {len(file_docs)} documents from {file_path}")
            except DocumentLoaderError as e:
                logger.warning(f"Skipping file {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {len(files)} JSON files")
        return documents

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
                raise DocumentLoaderError(f"Invalid document at index {i}: {e}") from e

        return documents

    def _dict_to_document(self, data: dict) -> Document:
        """Convert a single dictionary to a Document object.

        Args:
            data: Dictionary with document fields.

        Returns:
            Document object.

        Raises:
            KeyError: If required fields are missing.
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

    def _convert_document(self, lc_doc: LCDocument) -> Document:
        """Convert LangChain Document to core.models.Document.

        Args:
            lc_doc: LangChain document.

        Returns:
            Core Document object.

        Note:
            LangChain Document has:
                - page_content: str (the text)
                - metadata: dict (source, title, etc.)

            We map it to core.models.Document:
                - doc_id: generated from source or uuid
                - text: page_content
                - url: from metadata["source"] or empty
                - metadata: rest of metadata
        """
        # Generate doc_id from source path or create UUID
        source = lc_doc.metadata.get("source", "")
        doc_id = Path(source).stem if source else f"doc_{hash(lc_doc.page_content)}"

        # Extract URL from metadata (if present)
        url = lc_doc.metadata.get("url", lc_doc.metadata.get("source", ""))

        # Clean metadata (remove fields we've extracted)
        metadata = {k: v for k, v in lc_doc.metadata.items() if k not in ["source", "url"]}

        return Document(
            doc_id=doc_id,
            text=lc_doc.page_content,
            url=url,
            metadata=metadata,
        )
