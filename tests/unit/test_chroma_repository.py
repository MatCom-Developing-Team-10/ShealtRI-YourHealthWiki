"""Tests for ChromaRepository.

ChromaDB is an optional dependency in the test environment. If it isn't
installed, the entire module is skipped via ``importorskip``.
"""

from __future__ import annotations

import pytest

chromadb = pytest.importorskip("chromadb")

from core.models import Document
from infra.chroma_repository import ChromaRepository


@pytest.fixture
def repo(tmp_path) -> ChromaRepository:
    return ChromaRepository(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )


class TestAddAndSearch:
    def test_add_then_search_returns_added_doc(self, repo):
        docs = [
            Document("d1", "ignored text", "https://x"),
            Document("d2", "ignored text", "https://y"),
        ]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        repo.add_documents(docs, embeddings=embeddings)

        results = repo.search_similar([1.0, 0.0, 0.0], top_k=2)
        # d1 should rank first since its vector is identical
        assert len(results) >= 1
        assert results[0][0] == "d1"
        # Score is in [0, 1]
        for _, score in results:
            assert 0.0 <= score <= 1.0


class TestSearchSimilar:
    def test_returns_empty_on_empty_collection(self, repo):
        results = repo.search_similar([1.0, 0.0, 0.0], top_k=10)
        assert results == []
