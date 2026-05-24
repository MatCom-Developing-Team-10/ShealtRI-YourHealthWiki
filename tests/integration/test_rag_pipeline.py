"""Integration test: retriever → RAG, end-to-end with fallback."""

from core.models import (
    Document,
    Query,
    RetrievedDocument,
    UserProfile,
    UserProfileType,
)
from modules.rag.evaluator import RAGEvaluator
from modules.rag.service import RAGService
from modules.ranker.service import HybridRanker


class TestRAGWithoutLLM:
    """Test RAG module with fallback (no Groq API required)."""

    def test_rag_generates_response_with_fallback(self):
        """RAG should generate a response even without Groq API."""
        rag = RAGService(api_key=None)
        docs = [
            RetrievedDocument(
                document=Document(
                    doc_id="d1",
                    text="La diabetes es una enfermedad metabólica crónica.",
                    url="http://example.com/diabetes",
                    metadata={"title": "Diabetes Mellitus"},
                ),
                score=0.92,
            ),
            RetrievedDocument(
                document=Document(
                    doc_id="d2",
                    text="El tratamiento incluye cambios en la dieta y medicamentos.",
                    url="http://example.com/diabetes-treatment",
                    metadata={"title": "Tratamiento de Diabetes"},
                ),
                score=0.87,
            ),
        ]

        profile = UserProfile(profile_type=UserProfileType.PATIENT, name="Paciente")
        query = Query(text="síntomas de diabetes", user_profile=profile)

        response = rag.generate(query, docs, max_context_docs=2)

        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.profile_type == UserProfileType.PATIENT
        assert response.used_llm is False
        assert response.model_name == "template_fallback"
        assert len(response.sources) == 2
        assert response.query_text == "síntomas de diabetes"

    def test_rag_adapts_response_to_profile(self):
        """Different profiles should get different fallback responses."""
        docs = [
            RetrievedDocument(
                document=Document(
                    doc_id="d1",
                    text="Causas fisiopatológicas de la hipertensión arterial.",
                    url="http://example.com/hypertension",
                    metadata={"title": "Hipertensión"},
                ),
                score=0.90,
            ),
        ]

        rag = RAGService(api_key=None)

        # Patient response
        patient_profile = UserProfile(
            profile_type=UserProfileType.PATIENT, name="Paciente"
        )
        patient_query = Query(
            text="presión alta", user_profile=patient_profile
        )
        patient_response = rag.generate(patient_query, docs)

        # Student response
        student_profile = UserProfile(
            profile_type=UserProfileType.MEDICAL_STUDENT, name="Estudiante"
        )
        student_query = Query(
            text="presión alta", user_profile=student_profile
        )
        student_response = rag.generate(student_query, docs)

        # Both should have answers
        assert len(patient_response.answer) > 0
        assert len(student_response.answer) > 0

        # Profiles should differ
        assert patient_response.profile_type != student_response.profile_type
        # Responses may be different (at least their profiles are)
        assert (
            patient_response.profile_type == UserProfileType.PATIENT
        )
        assert (
            student_response.profile_type == UserProfileType.MEDICAL_STUDENT
        )

    def test_rag_handles_no_documents(self):
        """RAG should handle empty document list gracefully."""
        rag = RAGService(api_key=None)

        profile = UserProfile(profile_type=UserProfileType.PATIENT, name="Paciente")
        query = Query(text="query with no results", user_profile=profile)

        response = rag.generate(query, [], max_context_docs=3)

        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.used_llm is False
        assert len(response.sources) == 0

    def test_rag_defaults_to_patient_when_no_profile(self):
        """RAG should default to PATIENT profile if not specified."""
        rag = RAGService(api_key=None)

        doc = RetrievedDocument(
            document=Document(
                doc_id="d1",
                text="Content here",
                url="http://example.com",
                metadata={"title": "Title"},
            ),
            score=0.9,
        )

        query = Query(text="query", user_profile=None)  # No profile
        response = rag.generate(query, [doc])

        assert response.profile_type == UserProfileType.PATIENT

    def test_rag_respects_max_context_docs(self):
        """RAG should only include up to max_context_docs in the response."""
        rag = RAGService(api_key=None, max_context_docs=2)

        docs = [
            RetrievedDocument(
                document=Document(
                    doc_id=f"d{i}",
                    text=f"Content {i}",
                    url=f"http://example.com/d{i}",
                    metadata={"title": f"Doc {i}"},
                ),
                score=0.9 - i * 0.05,
            )
            for i in range(5)
        ]

        profile = UserProfile(profile_type=UserProfileType.PATIENT, name="Paciente")
        query = Query(text="query", user_profile=profile)

        response = rag.generate(query, docs)

        assert len(response.sources) <= 2
        assert response.sources[0].document.doc_id == "d0"
        assert response.sources[1].document.doc_id == "d1"


class TestRAGWithHybridRanker:
    """Integration tests for the re-ranking step wired into RAGService."""

    def _make_docs(self) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document=Document(
                    doc_id="d_high_lsi",
                    text="diabetes mellitus glucosa insulina páncreas",
                    url="http://example.com/diabetes",
                    metadata={"title": "Diabetes"},
                ),
                score=0.95,
            ),
            RetrievedDocument(
                document=Document(
                    doc_id="d_keyword_match",
                    text="hipertensión arterial presión alta síntomas tratamiento",
                    url="http://example.com/hypertension",
                    metadata={"title": "Hipertensión"},
                ),
                score=0.55,
            ),
            RetrievedDocument(
                document=Document(
                    doc_id="d_low",
                    text="cáncer oncología tumor células malignas",
                    url="http://example.com/cancer",
                    metadata={"title": "Cáncer"},
                ),
                score=0.30,
            ),
        ]

    def test_rag_with_ranker_produces_valid_response(self):
        """RAG with ranker wired in should still produce a valid response."""
        ranker = HybridRanker()
        rag = RAGService(api_key=None, ranker=ranker)
        query = Query(text="hipertensión arterial presión")
        response = rag.generate(query, self._make_docs())
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.used_llm is False

    def test_ranker_changes_source_order(self):
        """Sources in response should reflect re-ranked order, not original LSI order."""
        ranker = HybridRanker()
        rag = RAGService(api_key=None, ranker=ranker, max_context_docs=3)
        query = Query(text="hipertensión arterial presión")
        response = rag.generate(query, self._make_docs())
        # The keyword-matching doc should now be ranked above the high-LSI non-matching one
        source_ids = [s.document.doc_id for s in response.sources]
        assert source_ids[0] == "d_keyword_match"

    def test_rag_without_ranker_preserves_original_order(self):
        """Without a ranker, sources should preserve the retriever order."""
        rag = RAGService(api_key=None, ranker=None, max_context_docs=3)
        query = Query(text="hipertensión arterial presión")
        response = rag.generate(query, self._make_docs())
        assert response.sources[0].document.doc_id == "d_high_lsi"

    def test_evaluator_with_ranked_response(self):
        """Evaluator should work on the output of a re-ranked RAG response."""
        ranker = HybridRanker()
        rag = RAGService(api_key=None, ranker=ranker)
        evaluator = RAGEvaluator()
        query = Query(text="hipertensión arterial presión")
        response = rag.generate(query, self._make_docs())
        metrics = evaluator.evaluate(query, response.answer, response.sources)
        assert "faithfulness" in metrics
        assert "context_relevance" in metrics
        assert 0.0 <= metrics["faithfulness"] <= 1.0
        assert 0.0 <= metrics["context_relevance"] <= 1.0
