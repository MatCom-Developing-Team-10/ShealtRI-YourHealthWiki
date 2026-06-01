"""Tests for ``modules.evaluation.service._build_lsi_search_fn``.

This adapter converts the pipeline's chunk-level LSI retriever into the
document-level ``SearchFn`` contract used by ``EvaluationService``. The
chunk-to-document de-duplication and the optional ranker hook are the
non-trivial bits worth pinning.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.interfaces import BaseRanker
from core.models import Document, Query, RetrievedDocument
from modules.evaluation.service import _build_lsi_search_fn


def _chunk(doc_id: str, original: str | None, score: float = 0.5) -> RetrievedDocument:
    meta = {"original_doc_id": original} if original is not None else {}
    return RetrievedDocument(
        document=Document(doc_id=doc_id, text="", url="", metadata=meta),
        score=score,
    )


def _pipeline_with_results(results: list[RetrievedDocument]):
    """Build a SimpleNamespace exposing ``indexer`` and ``lsi`` like the real Pipeline."""
    indexer = MagicMock()
    indexer.build_query.return_value = MagicMock()  # opaque corpus

    lsi = MagicMock()
    lsi.retrieve.return_value = results

    return SimpleNamespace(indexer=indexer, lsi=lsi)


# ---------------------------------------------------------------------------
# Adapter behaviour
# ---------------------------------------------------------------------------


class TestSearchFnDeduplication:
    def test_extracts_original_doc_ids_in_order(self):
        results = [
            _chunk("doc_a__chunk_0", "doc_a"),
            _chunk("doc_b__chunk_0", "doc_b"),
            _chunk("doc_c__chunk_0", "doc_c"),
        ]
        search = _build_lsi_search_fn(_pipeline_with_results(results))
        assert search("query", 3) == ["doc_a", "doc_b", "doc_c"]

    def test_duplicate_original_ids_collapsed_preserving_rank(self):
        # Two chunks of doc_a then one chunk of doc_b → ranking is [a, b]
        results = [
            _chunk("doc_a__chunk_0", "doc_a", 0.9),
            _chunk("doc_a__chunk_1", "doc_a", 0.8),
            _chunk("doc_b__chunk_0", "doc_b", 0.7),
        ]
        search = _build_lsi_search_fn(_pipeline_with_results(results))
        assert search("q", 2) == ["doc_a", "doc_b"]

    def test_falls_back_to_doc_id_split_when_metadata_missing(self):
        # Chunks without 'original_doc_id' metadata fall back to splitting
        # on the '__chunk_' separator inside the doc_id itself.
        results = [
            _chunk("alpha__chunk_0", None),
            _chunk("alpha__chunk_3", None),
            _chunk("beta__chunk_2", None),
        ]
        search = _build_lsi_search_fn(_pipeline_with_results(results))
        assert search("q", 2) == ["alpha", "beta"]

    def test_empty_results_returns_empty(self):
        search = _build_lsi_search_fn(_pipeline_with_results([]))
        assert search("q", 5) == []


class TestSearchFnRanker:
    def test_ranker_called_when_provided(self):
        results = [
            _chunk("a__chunk_0", "a", 0.9),
            _chunk("b__chunk_0", "b", 0.7),
        ]

        class _ReversingRanker(BaseRanker):
            def __init__(self):
                self.called = False

            def rerank(self, query, retrieved):
                self.called = True
                return list(reversed(retrieved))

        ranker = _ReversingRanker()
        search = _build_lsi_search_fn(_pipeline_with_results(results), ranker=ranker)
        ordered = search("q", 2)

        assert ranker.called is True
        # The reranker reversed the chunk order, so the document order flips too
        assert ordered == ["b", "a"]

    def test_no_ranker_preserves_lsi_order(self):
        results = [
            _chunk("a__chunk_0", "a"),
            _chunk("b__chunk_0", "b"),
        ]
        search = _build_lsi_search_fn(_pipeline_with_results(results))
        assert search("q", 2) == ["a", "b"]


class TestSearchFnTopK:
    def test_requests_5x_top_k_from_lsi(self):
        pipeline = _pipeline_with_results([_chunk("a__chunk_0", "a")])
        search = _build_lsi_search_fn(pipeline)
        search("query text", 3)
        # The adapter over-fetches to absorb chunk de-duplication: top_k * 5
        pipeline.lsi.retrieve.assert_called_once()
        kwargs = pipeline.lsi.retrieve.call_args.kwargs
        assert kwargs["top_k"] == 15

    def test_query_indexed_corpus_is_set_from_indexer(self):
        pipeline = _pipeline_with_results([_chunk("a__chunk_0", "a")])
        search = _build_lsi_search_fn(pipeline)
        search("hipertensión arterial", 2)
        pipeline.indexer.build_query.assert_called_once_with("hipertensión arterial")
