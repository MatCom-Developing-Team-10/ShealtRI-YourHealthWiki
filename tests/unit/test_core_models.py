"""Tests for core.models — Document, Query, RetrievedDocument."""

from __future__ import annotations

import pytest

from core.models import Document, Query, RetrievedDocument


class TestDocument:
    def test_minimum_fields(self):
        doc = Document(doc_id="d1", text="hello", url="http://x")
        assert doc.doc_id == "d1"
        assert doc.text == "hello"
        assert doc.url == "http://x"
        assert doc.metadata == {}

    def test_metadata_default_is_independent_per_instance(self):
        a = Document(doc_id="a", text="", url="")
        b = Document(doc_id="b", text="", url="")
        a.metadata["k"] = 1
        assert "k" not in b.metadata

    def test_metadata_explicit(self):
        doc = Document(
            doc_id="d", text="t", url="u", metadata={"title": "T", "language": "es"}
        )
        assert doc.metadata["title"] == "T"
        assert doc.metadata["language"] == "es"


class TestQuery:
    def test_minimum_fields(self):
        q = Query(text="hipertensión")
        assert q.text == "hipertensión"
        assert q.indexed_corpus is None
        assert q.metadata == {}

    def test_metadata_default_is_independent_per_instance(self):
        a = Query(text="x")
        b = Query(text="y")
        a.metadata["k"] = 1
        assert "k" not in b.metadata


class TestRetrievedDocument:
    def test_pairs_document_and_score(self):
        doc = Document(doc_id="d", text="t", url="u")
        rd = RetrievedDocument(document=doc, score=0.87)
        assert rd.document is doc
        assert rd.score == pytest.approx(0.87)
