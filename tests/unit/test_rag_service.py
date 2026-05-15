"""Tests for RAGService — Gemini integration with fallback."""

from unittest.mock import MagicMock, patch

from core.models import (
    Document,
    Query,
    RetrievedDocument,
    UserProfile,
    UserProfileType,
)
from modules.rag.service import RAGService


def _make_retrieved(doc_id="d1", text="texto médico", score=0.9):
    doc = Document(
        doc_id=doc_id, text=text, url=f"http://x/{doc_id}", metadata={"title": doc_id}
    )
    return RetrievedDocument(document=doc, score=score)


def _make_query(text="diabetes síntomas", profile_type=UserProfileType.PATIENT):
    profile = UserProfile(profile_type=profile_type, name="Paciente")
    return Query(text=text, user_profile=profile)


class TestRAGServiceFallback:
    """Test template fallback when Gemini is unavailable."""

    def test_fallback_when_no_api_key(self):
        """Service without API key should use fallback."""
        service = RAGService(api_key=None)
        query = _make_query()
        docs = [_make_retrieved()]
        response = service.generate(query, docs)
        assert response.used_llm is False
        assert response.model_name == "template_fallback"
        assert len(response.answer) > 0

    def test_fallback_preserves_sources(self):
        service = RAGService(api_key=None)
        docs = [_make_retrieved(f"d{i}") for i in range(3)]
        query = _make_query()
        response = service.generate(query, docs)
        assert len(response.sources) <= 3

    def test_fallback_answer_adapts_to_profile(self):
        service = RAGService(api_key=None)
        docs = [_make_retrieved()]

        response_patient = service.generate(
            _make_query(profile_type=UserProfileType.PATIENT), docs
        )
        response_student = service.generate(
            _make_query(profile_type=UserProfileType.MEDICAL_STUDENT), docs
        )

        assert response_patient.profile_type == UserProfileType.PATIENT
        assert response_student.profile_type == UserProfileType.MEDICAL_STUDENT
        # Answers may differ based on profile
        assert len(response_patient.answer) > 0
        assert len(response_student.answer) > 0


class TestRAGServiceGeminiSuccess:
    """Test happy path with mocked Gemini response."""

    def test_generate_uses_llm_when_available(self):
        """Mock Gemini to return a successful response."""
        with patch("modules.rag.service.os.environ.get") as mock_env:
            mock_env.return_value = "fake-api-key"

            with patch("modules.rag.service.genai") as mock_genai:
                mock_model = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "La diabetes es una enfermedad metabólica..."
                mock_model.generate_content.return_value = mock_response
                mock_genai.GenerativeModel.return_value = mock_model

                service = RAGService(api_key="fake-api-key")
                service._client = mock_genai
                query = _make_query()
                docs = [_make_retrieved()]
                response = service.generate(query, docs)

        assert response.used_llm is True
        assert "diabetes" in response.answer.lower() or len(response.answer) > 0
        assert response.model_name == "gemini-1.5-flash"

    def test_generate_with_no_user_profile_defaults_to_patient(self):
        service = RAGService(api_key=None)
        query = Query(text="fiebre", user_profile=None)
        response = service.generate(query, [_make_retrieved()])

        assert response.profile_type == UserProfileType.PATIENT

    def test_respects_max_context_docs(self):
        service = RAGService(api_key=None, max_context_docs=2)
        query = _make_query()
        docs = [_make_retrieved(f"d{i}") for i in range(5)]
        response = service.generate(query, docs)

        assert len(response.sources) <= 2


class TestRAGServiceAvailability:
    def test_is_available_returns_false_without_api_key(self):
        service = RAGService(api_key=None)
        assert service.is_available() is False

    def test_is_available_returns_true_with_api_key_and_client(self):
        with patch("modules.rag.service.os.environ.get") as mock_env:
            mock_env.return_value = "fake-key"

            with patch("modules.rag.service.genai") as mock_genai:
                service = RAGService(api_key="fake-key")
                service._client = mock_genai
                assert service.is_available() is True


class TestRAGServiceProfileResolution:
    def test_resolve_profile_with_none_defaults_to_patient(self):
        service = RAGService(api_key=None)
        profile = service._resolve_profile(None)
        assert profile.profile_type == UserProfileType.PATIENT
        assert profile.name == "Paciente"

    def test_resolve_profile_returns_provided_profile(self):
        service = RAGService(api_key=None)
        provided = UserProfile(
            profile_type=UserProfileType.MEDICAL_STUDENT, name="Estudiante"
        )
        profile = service._resolve_profile(provided)
        assert profile.profile_type == UserProfileType.MEDICAL_STUDENT
        assert profile.name == "Estudiante"
