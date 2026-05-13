"""Tests for core.interfaces — IndexedCorpus invariants and ABC contracts."""

from __future__ import annotations

import pytest

from core.interfaces import (
    BaseRepository,
    BaseRetriever,
    DocumentStore,
    IndexedCorpus,
)
from core.models import Document, Query


class TestIndexedCorpus:
    def test_valid_construction(self):
        docs = [Document("a", "t1", "u1"), Document("b", "t2", "u2")]
        corpus = IndexedCorpus(
            documents=docs,
            processed_texts=["t1", "t2"],
            inverted_index={"x": [(0, 1)]},
            vocabulary=["x"],
        )
        assert len(corpus) == 2
        assert corpus.doc_ids == ["a", "b"]

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="must have the same length"):
            IndexedCorpus(
                documents=[Document("a", "", "")],
                processed_texts=["x", "y"],  # mismatched
                inverted_index={},
                vocabulary=[],
            )

    def test_empty_corpus_is_valid(self):
        corpus = IndexedCorpus(
            documents=[], processed_texts=[], inverted_index={}, vocabulary=[]
        )
        assert len(corpus) == 0
        assert corpus.doc_ids == []


class TestABCContracts:
    """The abstract base classes must reject instantiation when methods missing."""

    def test_document_store_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DocumentStore()  # type: ignore[abstract]

    def test_base_repository_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseRepository()  # type: ignore[abstract]

    def test_base_retriever_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseRetriever()  # type: ignore[abstract]

    def test_partial_subclass_still_abstract(self):
        class PartialStore(DocumentStore):
            def add_documents(self, documents):
                pass
            # Missing get_by_id and get_by_ids

        with pytest.raises(TypeError):
            PartialStore()  # type: ignore[abstract]

    def test_complete_subclass_works(self):
        class GoodStore(DocumentStore):
            def add_documents(self, documents):
                pass

            def get_by_id(self, doc_id):
                return None

            def get_by_ids(self, doc_ids):
                return []

        instance = GoodStore()
        assert instance.exists("anything") is False  # default impl
