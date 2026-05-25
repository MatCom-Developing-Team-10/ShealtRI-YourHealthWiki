"""Tests for RAGEvaluator — faithfulness and context relevance metrics."""

from __future__ import annotations

from core.models import Document, Query, RetrievedDocument
from modules.rag.evaluator import RAGEvaluator


def _doc(doc_id: str, text: str, score: float = 0.7) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(doc_id=doc_id, text=text, url=f"http://x/{doc_id}"),
        score=score,
    )


def _query(text: str = "diabetes síntomas") -> Query:
    return Query(text=text)


class TestFaithfulness:
    def test_answer_fully_grounded_in_context(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "diabetes mellitus glucosa insulina páncreas")]
        answer = "diabetes glucosa insulina"
        score = evaluator.faithfulness(answer, context)
        assert score > 0.5

    def test_answer_not_in_context_returns_low_score(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "diabetes mellitus glucosa insulina")]
        answer = "astronomía galaxias nebulosa universo telescopio"
        score = evaluator.faithfulness(answer, context)
        assert score < 0.3

    def test_empty_answer_returns_zero(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "texto médico")]
        assert evaluator.faithfulness("", context) == 0.0

    def test_empty_context_returns_zero(self):
        evaluator = RAGEvaluator()
        assert evaluator.faithfulness("respuesta médica", []) == 0.0

    def test_score_in_zero_one_range(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "hipertensión arterial presión alta tratamiento")]
        answer = "la hipertensión arterial requiere tratamiento médico adecuado"
        score = evaluator.faithfulness(answer, context)
        assert 0.0 <= score <= 1.0

    def test_multiple_context_docs(self):
        evaluator = RAGEvaluator()
        context = [
            _doc("d1", "hipertensión síntomas dolor cabeza"),
            _doc("d2", "presión arterial medicamentos tratamiento"),
        ]
        answer = "hipertensión presión medicamentos"
        score = evaluator.faithfulness(answer, context)
        assert score > 0.3


class TestContextRelevance:
    def test_high_scores_produce_high_relevance(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "texto", score=0.9), _doc("d2", "texto", score=0.8)]
        assert evaluator.context_relevance(context) == pytest.approx(0.85)

    def test_empty_context_returns_zero(self):
        evaluator = RAGEvaluator()
        assert evaluator.context_relevance([]) == 0.0

    def test_single_document(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "texto", score=0.6)]
        assert evaluator.context_relevance(context) == pytest.approx(0.6)

    def test_score_is_mean_of_retriever_scores(self):
        evaluator = RAGEvaluator()
        scores = [0.4, 0.6, 0.8]
        context = [_doc(f"d{i}", "texto", score=s) for i, s in enumerate(scores)]
        expected = sum(scores) / len(scores)
        assert evaluator.context_relevance(context) == pytest.approx(expected)


class TestGroundedness:
    def test_fully_grounded_answer_scores_high(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "la hipertensión arterial causa presión alta en las arterias")]
        answer = "La hipertensión arterial causa presión alta."
        score = evaluator.groundedness(answer, context)
        assert score >= 0.5

    def test_hallucinated_answer_scores_low(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "diabetes glucosa insulina páncreas")]
        answer = "Los astronautas viajan al espacio con cohetes modernos."
        score = evaluator.groundedness(answer, context)
        assert score < 0.3

    def test_empty_answer_returns_zero(self):
        evaluator = RAGEvaluator()
        assert evaluator.groundedness("", [_doc("d1", "texto médico")]) == 0.0

    def test_empty_context_returns_zero(self):
        evaluator = RAGEvaluator()
        assert evaluator.groundedness("respuesta médica sobre diabetes", []) == 0.0

    def test_score_in_zero_one_range(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "hipertensión arterial presión alta tratamiento médico")]
        answer = "La hipertensión requiere tratamiento. Los síntomas incluyen presión alta."
        score = evaluator.groundedness(answer, context)
        assert 0.0 <= score <= 1.0

    def test_partial_support_returns_intermediate_score(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "diabetes glucosa insulina tratamiento")]
        # Two sentences: one supported, one not
        answer = "La diabetes requiere glucosa controlada. Los cohetes viajan al espacio exterior."
        score = evaluator.groundedness(answer, context)
        assert 0.0 < score < 1.0


class TestEvaluate:
    def test_evaluate_returns_all_three_metrics(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "diabetes glucosa insulina tratamiento")]
        answer = "diabetes glucosa tratamiento"
        result = evaluator.evaluate(_query(), answer, context)
        assert "faithfulness" in result
        assert "groundedness" in result
        assert "context_relevance" in result

    def test_evaluate_values_in_range(self):
        evaluator = RAGEvaluator()
        context = [_doc("d1", "hipertensión arterial presión", score=0.75)]
        answer = "la hipertensión arterial es una enfermedad crónica"
        result = evaluator.evaluate(_query("hipertensión"), answer, context)
        assert 0.0 <= result["faithfulness"] <= 1.0
        assert 0.0 <= result["groundedness"] <= 1.0
        assert 0.0 <= result["context_relevance"] <= 1.0


import pytest
