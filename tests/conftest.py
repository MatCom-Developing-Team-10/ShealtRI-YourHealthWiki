"""Shared pytest fixtures for the ShealtRI test suite."""

from __future__ import annotations

import pytest

from core.models import Document
from modules.text_processor.service import TextProcessor
from tests._synthetic_corpus import RAW_DOCUMENTS


@pytest.fixture(scope="session")
def sample_documents() -> list[Document]:
    """Twenty synthetic Spanish medical documents for pipeline testing."""
    return [
        Document(
            doc_id=d["doc_id"],
            text=d["text"],
            url=d["url"],
            metadata={"title": d["title"]},
        )
        for d in RAW_DOCUMENTS
    ]


@pytest.fixture(scope="session")
def text_processor() -> TextProcessor:
    """Shared TextProcessor instance (spaCy model loaded once per session)."""
    return TextProcessor()


@pytest.fixture
def tmp_chroma_dir(tmp_path) -> str:
    """Temporary directory for ChromaDB — isolated per test."""
    return str(tmp_path / "chroma")


@pytest.fixture
def tmp_store_dir(tmp_path) -> str:
    """Temporary directory for FileSystemDocumentStore — isolated per test."""
    return str(tmp_path / "store")
