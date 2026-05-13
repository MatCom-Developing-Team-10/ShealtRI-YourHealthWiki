"""Shared pytest fixtures for the ShealtRI test suite.

Conventions
-----------
- ``tmp_path`` (pytest builtin) is used for every disk-touching test so the
  test process never writes outside its sandbox.
- Heavy-import fixtures (TextProcessor, which loads spaCy) are scoped at
  the session level so the spaCy model is loaded only once per test run.
- ABC-based interfaces (BaseRepository, DocumentStore) are mocked via
  small in-memory fakes defined here, keeping individual test files focused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.interfaces import BaseRepository, DocumentStore
from core.models import Document
from tests._synthetic_corpus import RAW_DOCUMENTS


# ---------------------------------------------------------------------------
# In-memory fakes for the data-layer ABCs
# ---------------------------------------------------------------------------

class InMemoryDocumentStore(DocumentStore):
    """Minimal in-memory DocumentStore used to keep retriever tests fast."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    def add_documents(self, documents: list[Document]) -> None:
        for doc in documents:
            self._docs[doc.doc_id] = doc

    def get_by_id(self, doc_id: str):
        return self._docs.get(doc_id)

    def get_by_ids(self, doc_ids: list[str]) -> list[Document]:
        return [self._docs[d] for d in doc_ids if d in self._docs]

    def exists(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def delete(self, doc_id: str) -> bool:
        return self._docs.pop(doc_id, None) is not None


class InMemoryRepository(BaseRepository):
    """Vector repository fake using simple cosine similarity over Python lists."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._metas: list[dict] = []

    def add_documents(self, documents, embeddings=None) -> None:
        for i, doc in enumerate(documents):
            self._ids.append(doc.doc_id)
            self._metas.append({"url": doc.url})
            if embeddings is not None:
                self._vectors.append(list(embeddings[i]))
            else:
                self._vectors.append([])

    def search_similar(self, query_vector, top_k=10):
        import math

        def cos(a, b):
            num = sum(x * y for x, y in zip(a, b))
            da = math.sqrt(sum(x * x for x in a))
            db = math.sqrt(sum(y * y for y in b))
            if da == 0 or db == 0:
                return 0.0
            return num / (da * db)

        scored = [
            (doc_id, cos(query_vector, vec))
            for doc_id, vec in zip(self._ids, self._vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()


@pytest.fixture
def in_memory_repo() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture(scope="session")
def sample_documents() -> list[Document]:
    """Twenty synthetic Spanish medical documents for pipeline/integration testing."""
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
def text_processor():
    """Real TextProcessor instance, shared across the session.

    Loading the spaCy model is expensive (~1s), so a session-scoped fixture
    keeps the test suite fast. Tests that need a fresh spell-checker
    vocabulary should create their own TextProcessor inside the test.
    """
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("es_core_news_md")
    except OSError:
        pytest.skip("spaCy model 'es_core_news_md' not installed")

    from modules.text_processor import TextProcessor

    return TextProcessor()


@pytest.fixture
def tmp_chroma_dir(tmp_path) -> str:
    """Temporary directory for ChromaDB — isolated per test."""
    return str(tmp_path / "chroma")


@pytest.fixture
def tmp_store_dir(tmp_path) -> str:
    """Temporary directory for FileSystemDocumentStore — isolated per test."""
    return str(tmp_path / "store")
