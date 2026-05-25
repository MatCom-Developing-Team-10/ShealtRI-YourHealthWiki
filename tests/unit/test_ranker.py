"""Tests for HybridRanker — BM25 + LSI re-ranking."""

from __future__ import annotations

import pytest

from core.models import Document, Query, RetrievedDocument, UserProfile, UserProfileType
from modules.ranker.service import HybridRanker


def _doc(doc_id: str, text: str, score: float = 0.5) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(doc_id=doc_id, text=text, url=f"http://x/{doc_id}"),
        score=score,
    )


def _query(text: str = "hipertensión síntomas") -> Query:
    return Query(text=text)


class TestHybridRankerInit:
    def test_default_weights(self):
        ranker = HybridRanker()
        assert ranker._lsi_weight == 0.6
        assert ranker._bm25_weight == 0.4

    def test_custom_weights_sum_to_one(self):
        ranker = HybridRanker(lsi_weight=0.7, bm25_weight=0.3)
        assert ranker._lsi_weight == 0.7
        assert ranker._bm25_weight == 0.3

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError):
            HybridRanker(lsi_weight=0.5, bm25_weight=0.3)


class TestHybridRankerRerank:
    def test_empty_list_returns_empty(self):
        ranker = HybridRanker()
        result = ranker.rerank(_query(), [])
        assert result == []

    def test_single_document_returns_without_error(self):
        ranker = HybridRanker()
        docs = [_doc("d1", "diabetes mellitus symptoms treatment", score=0.8)]
        result = ranker.rerank(_query("diabetes"), docs)
        assert len(result) == 1
        assert result[0].document.doc_id == "d1"

    def test_returns_same_number_of_documents(self):
        ranker = HybridRanker()
        docs = [_doc(f"d{i}", f"medical text about topic {i}", score=0.5 + i * 0.1) for i in range(5)]
        result = ranker.rerank(_query(), docs)
        assert len(result) == 5

    def test_scores_in_zero_one_range(self):
        ranker = HybridRanker()
        docs = [
            _doc("d1", "hipertensión arterial presión alta síntomas", score=0.9),
            _doc("d2", "diabetes mellitus tipo 2 glucosa", score=0.3),
            _doc("d3", "cáncer tumor oncología tratamiento", score=0.6),
        ]
        result = ranker.rerank(_query("hipertensión síntomas presión"), docs)
        for doc in result:
            assert 0.0 <= doc.score <= 1.0

    def test_lexically_relevant_document_ranks_higher(self):
        """BM25 signal should elevate a keyword-matching doc above a semantically-only peer.

        With three documents, d_anchor has the highest LSI score and wins overall.
        d_lexical has a slightly lower LSI than d_semantic but exact query term matches,
        so BM25 should push it above d_semantic — demonstrating that the ranker is not
        purely ordered by LSI score.
        """
        ranker = HybridRanker()
        query = _query("hipertensión arterial presión")
        docs = [
            # Highest LSI — anchor doc that "wins" overall
            _doc("d_anchor", "cardiovascular riesgo tratamiento médico", score=0.90),
            # Mid LSI, no query term overlap
            _doc("d_semantic", "diabetes mellitus glucosa insulina", score=0.70),
            # Lowest LSI but exact query term match — BM25 should push it above d_semantic
            _doc("d_lexical", "hipertensión arterial presión alta tratamiento", score=0.65),
        ]
        result = ranker.rerank(query, docs)
        ids = [d.document.doc_id for d in result]
        lexical_pos = ids.index("d_lexical")
        semantic_pos = ids.index("d_semantic")
        assert lexical_pos < semantic_pos, (
            "d_lexical should rank above d_semantic after hybrid re-rank "
            f"(got positions lexical={lexical_pos}, semantic={semantic_pos})"
        )

    def test_result_is_sorted_descending(self):
        ranker = HybridRanker()
        docs = [_doc(f"d{i}", f"texto sobre salud {i}", score=float(i) / 10) for i in range(6)]
        result = ranker.rerank(_query("salud"), docs)
        scores = [d.score for d in result]
        assert scores == sorted(scores, reverse=True)

    def test_all_input_doc_ids_present_in_output(self):
        ranker = HybridRanker()
        docs = [_doc(f"d{i}", f"medical text {i}", score=0.5) for i in range(4)]
        result = ranker.rerank(_query(), docs)
        input_ids = {d.document.doc_id for d in docs}
        output_ids = {d.document.doc_id for d in result}
        assert input_ids == output_ids

    def test_identical_lsi_scores_no_crash(self):
        """All docs with the same LSI score should not cause division by zero."""
        ranker = HybridRanker()
        docs = [_doc(f"d{i}", f"texto médico {i}", score=0.5) for i in range(3)]
        result = ranker.rerank(_query(), docs)
        assert len(result) == 3
