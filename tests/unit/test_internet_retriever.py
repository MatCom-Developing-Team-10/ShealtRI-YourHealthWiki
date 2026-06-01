"""Tests for ``modules.web_search.internet_retriever.InternetSearchRetriever``.

All network I/O is mocked via a fake ``WebContentFetcher`` so the tests
are deterministic and never touch the live internet.
"""

from __future__ import annotations

import pytest

from core.interfaces import DocumentStore
from core.models import Document, Query
from modules.web_search.internet_retriever import InternetSearchRetriever


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeFetcher:
    """Fetcher stub returning a pre-configured list of documents per call."""

    def __init__(self, results: list[Document]):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def fetch(self, query: str, max_results: int = 5) -> list[Document]:
        self.calls.append((query, max_results))
        return list(self.results[:max_results])


class _FakeStore(DocumentStore):
    def __init__(self):
        self.docs: dict[str, Document] = {}

    def add_documents(self, documents):
        for d in documents:
            self.docs[d.doc_id] = d

    def get_by_id(self, doc_id):
        return self.docs.get(doc_id)

    def get_by_ids(self, doc_ids):
        return [self.docs[i] for i in doc_ids if i in self.docs]

    def exists(self, doc_id):
        return doc_id in self.docs


def _doc(doc_id: str) -> Document:
    return Document(doc_id=doc_id, text="x" * 300, url=f"https://example.org/{doc_id}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrieveBasic:
    def test_empty_query_returns_empty_list(self):
        retriever = InternetSearchRetriever(_FakeFetcher([]))
        out = retriever.retrieve(Query(text=""), top_k=5)
        assert out == []

    def test_whitespace_query_returns_empty_list(self):
        retriever = InternetSearchRetriever(_FakeFetcher([]))
        assert retriever.retrieve(Query(text="   "), top_k=5) == []

    def test_no_results_returns_empty(self):
        retriever = InternetSearchRetriever(_FakeFetcher([]))
        assert retriever.retrieve(Query(text="hipertensión"), top_k=5) == []

    def test_fetcher_called_with_query_text_and_top_k(self):
        fetcher = _FakeFetcher([_doc("a"), _doc("b")])
        retriever = InternetSearchRetriever(fetcher)
        retriever.retrieve(Query(text="diabetes"), top_k=3)
        assert fetcher.calls == [("diabetes", 3)]


class TestScoring:
    def test_first_result_scores_highest(self):
        fetcher = _FakeFetcher([_doc("a"), _doc("b"), _doc("c")])
        retriever = InternetSearchRetriever(fetcher)
        out = retriever.retrieve(Query(text="q"), top_k=5)
        assert out[0].score > out[1].score > out[2].score

    def test_scores_in_unit_interval(self):
        fetcher = _FakeFetcher([_doc(f"d{i}") for i in range(5)])
        retriever = InternetSearchRetriever(fetcher)
        out = retriever.retrieve(Query(text="q"), top_k=5)
        for r in out:
            assert 0.0 <= r.score <= 1.0

    def test_first_score_is_one_minus_zero_over_n_plus_one(self):
        fetcher = _FakeFetcher([_doc(f"d{i}") for i in range(4)])
        retriever = InternetSearchRetriever(fetcher)
        out = retriever.retrieve(Query(text="q"), top_k=4)
        # rank 0 / (4+1) = 0 → score = 1.0
        assert out[0].score == pytest.approx(1.0)


class TestCaching:
    def test_documents_are_cached_when_store_provided(self):
        fetcher = _FakeFetcher([_doc("a"), _doc("b")])
        store = _FakeStore()
        retriever = InternetSearchRetriever(fetcher, document_store=store)
        retriever.retrieve(Query(text="q"), top_k=5)
        assert "a" in store.docs
        assert "b" in store.docs

    def test_duplicates_are_skipped(self):
        fetcher = _FakeFetcher([_doc("a"), _doc("b")])
        store = _FakeStore()
        store.docs["a"] = _doc("a")
        retriever = InternetSearchRetriever(fetcher, document_store=store)
        retriever.retrieve(Query(text="q"), top_k=5)
        # 'b' is new and added; 'a' was already there
        assert "b" in store.docs

    def test_cache_failure_is_swallowed(self):
        class _CrashingStore(_FakeStore):
            def add_documents(self, documents):
                raise RuntimeError("disk on fire")

        retriever = InternetSearchRetriever(
            _FakeFetcher([_doc("a")]),
            document_store=_CrashingStore(),
        )
        # Should not raise — just logs a warning
        out = retriever.retrieve(Query(text="q"), top_k=5)
        assert len(out) == 1

    def test_no_store_means_no_caching_attempt(self):
        # Just ensure the code path without a store is exercised
        retriever = InternetSearchRetriever(_FakeFetcher([_doc("a")]))
        assert len(retriever.retrieve(Query(text="q"), top_k=5)) == 1
