"""Tests for ``modules.retriever.fallback_retriever.FallbackRetriever``.

The fallback uses two ``BaseRetriever`` instances. We inject in-memory
stubs so the tests are deterministic and have no I/O.
"""

from __future__ import annotations

import pytest

from core.interfaces import BaseRetriever
from core.models import Document, Query, RetrievedDocument
from modules.retriever.fallback_retriever import FallbackRetriever


def _doc(doc_id: str) -> Document:
    return Document(doc_id=doc_id, text=doc_id, url=f"http://x/{doc_id}")


def _rd(doc_id: str, score: float) -> RetrievedDocument:
    return RetrievedDocument(document=_doc(doc_id), score=score)


class _StubRetriever(BaseRetriever):
    """Returns a pre-configured ranked list, regardless of the query."""

    def __init__(self, results: list[RetrievedDocument], *, raise_on_call: bool = False):
        self.results = results
        self.raise_on_call = raise_on_call
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: Query, top_k: int = 10):
        self.calls.append((query.text, top_k))
        if self.raise_on_call:
            raise RuntimeError("primary failed")
        return self.results[:top_k]


# ---------------------------------------------------------------------------
# Phase-1 only: primary returns enough
# ---------------------------------------------------------------------------


class TestPrimaryOnly:
    def test_fallback_skipped_when_primary_meets_min_results(self):
        primary = _StubRetriever([_rd("a", 0.9), _rd("b", 0.8), _rd("c", 0.7)])
        fallback = _StubRetriever([_rd("x", 0.5)])
        retriever = FallbackRetriever(primary=primary, fallback=fallback, min_results=3)

        out = retriever.retrieve(Query(text="q"), top_k=5)
        assert [r.document.doc_id for r in out] == ["a", "b", "c"]
        assert fallback.calls == []   # fallback never called

    def test_top_k_truncates_primary_results(self):
        primary = _StubRetriever(
            [_rd("a", 0.9), _rd("b", 0.8), _rd("c", 0.7), _rd("d", 0.6)]
        )
        fallback = _StubRetriever([])
        retriever = FallbackRetriever(primary=primary, fallback=fallback, min_results=1)
        out = retriever.retrieve(Query(text="q"), top_k=2)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Phase-2: fallback merges
# ---------------------------------------------------------------------------


class TestFallbackActivation:
    def test_fallback_runs_when_primary_below_min_results(self):
        primary = _StubRetriever([_rd("a", 0.9)])  # only 1
        fallback = _StubRetriever([_rd("x", 0.5), _rd("y", 0.4)])
        retriever = FallbackRetriever(primary=primary, fallback=fallback, min_results=3)

        out = retriever.retrieve(Query(text="q"), top_k=5)
        ids = [r.document.doc_id for r in out]
        # Primary first (highest score), then fallbacks sorted by score
        assert "a" in ids
        assert "x" in ids
        assert "y" in ids
        # Sorted by score descending
        scores = [r.score for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_merged_top_k_respected(self):
        primary = _StubRetriever([_rd("a", 0.9)])
        fallback = _StubRetriever(
            [_rd("x", 0.5), _rd("y", 0.4), _rd("z", 0.3)]
        )
        retriever = FallbackRetriever(primary=primary, fallback=fallback, min_results=3)
        out = retriever.retrieve(Query(text="q"), top_k=2)
        assert len(out) == 2

    def test_deduplication_keeps_higher_score(self):
        # Same doc 'a' returned by both — primary score 0.9, fallback 0.4.
        primary = _StubRetriever([_rd("a", 0.9)])
        fallback = _StubRetriever([_rd("a", 0.4), _rd("y", 0.3)])
        retriever = FallbackRetriever(primary=primary, fallback=fallback, min_results=3)

        out = retriever.retrieve(Query(text="q"), top_k=5)
        doc_a = next(r for r in out if r.document.doc_id == "a")
        assert doc_a.score == 0.9   # primary score preserved

    def test_dedup_keeps_higher_score_even_when_fallback_wins(self):
        # Edge case: fallback has a strictly higher score for the same doc
        primary = _StubRetriever([_rd("a", 0.3)])
        fallback = _StubRetriever([_rd("a", 0.9), _rd("y", 0.2)])
        retriever = FallbackRetriever(primary=primary, fallback=fallback, min_results=3)
        out = retriever.retrieve(Query(text="q"), top_k=5)
        doc_a = next(r for r in out if r.document.doc_id == "a")
        assert doc_a.score == 0.9


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_primary_exception_treated_as_empty(self):
        # Primary raises → primary results = []; min_results=3 ⇒ fallback runs
        primary = _StubRetriever([], raise_on_call=True)
        fallback = _StubRetriever([_rd("x", 0.5), _rd("y", 0.4), _rd("z", 0.3)])
        retriever = FallbackRetriever(primary=primary, fallback=fallback, min_results=3)
        out = retriever.retrieve(Query(text="q"), top_k=5)
        assert {r.document.doc_id for r in out} == {"x", "y", "z"}

    def test_fallback_exception_returns_primary_only(self):
        primary = _StubRetriever([_rd("a", 0.9)])
        fallback = _StubRetriever([], raise_on_call=True)
        retriever = FallbackRetriever(primary=primary, fallback=fallback, min_results=3)
        out = retriever.retrieve(Query(text="q"), top_k=5)
        assert [r.document.doc_id for r in out] == ["a"]


# ---------------------------------------------------------------------------
# fit delegation
# ---------------------------------------------------------------------------


class TestFitDelegation:
    def test_fit_calls_primary_not_fallback(self):
        class _FittableStub(BaseRetriever):
            def __init__(self):
                self.fit_called = False
                self.fit_arg = None

            def fit(self, corpus):
                self.fit_called = True
                self.fit_arg = corpus

            def retrieve(self, query, top_k=10):
                return []

        primary = _FittableStub()
        fallback = _StubRetriever([])  # has no fit
        retriever = FallbackRetriever(primary=primary, fallback=fallback)

        corpus_sentinel = object()
        retriever.fit(corpus_sentinel)
        assert primary.fit_called
        assert primary.fit_arg is corpus_sentinel
