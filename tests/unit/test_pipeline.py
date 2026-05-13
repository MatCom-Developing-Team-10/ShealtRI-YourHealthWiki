"""Tests for core.pipeline.RetrievalContext (Strategy Pattern)."""

from __future__ import annotations

from core.interfaces import BaseRetriever
from core.models import Document, Query, RetrievedDocument
from core.pipeline import RetrievalContext


class _FakeRetriever(BaseRetriever):
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: Query, top_k: int = 10):
        self.calls.append((query.text, top_k))
        return [
            RetrievedDocument(
                document=Document(doc_id=f"{self.label}", text=query.text, url=""),
                score=1.0,
            )
        ]


class TestRetrievalContext:
    def test_executes_strategy(self):
        strat = _FakeRetriever("A")
        ctx = RetrievalContext(strategy=strat)
        results = ctx.execute_search(Query(text="hello"), top_k=5)
        assert strat.calls == [("hello", 5)]
        assert len(results) == 1
        assert results[0].document.doc_id == "A"

    def test_strategy_swap_at_runtime(self):
        first = _FakeRetriever("first")
        second = _FakeRetriever("second")
        ctx = RetrievalContext(strategy=first)
        ctx.execute_search(Query(text="q1"))
        assert ctx.strategy is first

        ctx.strategy = second
        ctx.execute_search(Query(text="q2"))
        assert ctx.strategy is second
        assert first.calls == [("q1", 10)]
        assert second.calls == [("q2", 10)]

    def test_default_top_k_is_ten(self):
        strat = _FakeRetriever("X")
        ctx = RetrievalContext(strategy=strat)
        ctx.execute_search(Query(text="any"))
        assert strat.calls[-1][1] == 10
