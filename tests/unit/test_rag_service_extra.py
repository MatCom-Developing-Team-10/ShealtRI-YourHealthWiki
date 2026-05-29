"""Additional coverage for ``modules.rag.service.RAGService``.

Focuses on paths the main ``test_rag_service.py`` skips when groq is not
installed: profile resolution, ranker integration, and the LLM-call
fallback wiring through a fake Groq client injected directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.interfaces import BaseRanker
from core.models import (
    Document,
    Query,
    RetrievedDocument,
    UserProfile,
    UserProfileType,
)
from modules.rag.service import RAGService


def _rd(doc_id: str = "d1") -> RetrievedDocument:
    doc = Document(
        doc_id=doc_id,
        text="contenido médico relevante",
        url=f"http://example/{doc_id}",
        metadata={"title": doc_id},
    )
    return RetrievedDocument(document=doc, score=0.9)


def _q(text: str = "diabetes", profile: UserProfileType | None = UserProfileType.PATIENT):
    if profile is None:
        return Query(text=text, user_profile=None)
    return Query(text=text, user_profile=UserProfile(profile_type=profile, name="X"))


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


class TestProfileResolution:
    def test_none_profile_defaults_to_patient(self):
        out = RAGService._resolve_profile(None)
        assert out.profile_type == UserProfileType.PATIENT

    def test_existing_profile_passed_through(self):
        profile = UserProfile(profile_type=UserProfileType.MEDICAL_PROFESSIONAL, name="Dra. X")
        assert RAGService._resolve_profile(profile) is profile


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_false_without_api_key(self):
        assert RAGService(api_key=None).is_available() is False

    def test_false_when_client_init_fails(self, monkeypatch):
        # api key set, but Groq import fails → client stays None
        service = RAGService(api_key=None)
        service.api_key = "fake"
        service._client = None
        assert service.is_available() is False


# ---------------------------------------------------------------------------
# _call_llm fallback paths (work without groq installed)
# ---------------------------------------------------------------------------


class TestCallLLMFallback:
    def test_no_client_returns_fallback_marker(self):
        service = RAGService(api_key=None)
        answer, used_llm, model = service._call_llm("any prompt")
        assert used_llm is False
        assert model == "template_fallback"
        assert answer == ""

    def test_client_exception_falls_back(self):
        service = RAGService(api_key=None)
        service._client = MagicMock()
        service._client.chat.completions.create.side_effect = RuntimeError("rate limited")

        answer, used_llm, model = service._call_llm("prompt")
        assert used_llm is False
        assert model == "template_fallback"

    def test_empty_llm_response_falls_back(self):
        service = RAGService(api_key=None)
        service._client = MagicMock()
        # Groq returns an empty string
        mock_choice = MagicMock()
        mock_choice.message.content = "   "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        service._client.chat.completions.create.return_value = mock_response

        answer, used_llm, model = service._call_llm("prompt")
        assert used_llm is False
        assert model == "template_fallback"

    def test_successful_response_returns_text(self):
        service = RAGService(api_key=None, model_name="test-model")
        service._client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Una respuesta del modelo."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        service._client.chat.completions.create.return_value = mock_response

        answer, used_llm, model = service._call_llm("prompt")
        assert used_llm is True
        assert "respuesta" in answer
        assert model == "test-model"


# ---------------------------------------------------------------------------
# generate() with no LLM available — pure fallback path
# ---------------------------------------------------------------------------


class TestGenerateFallback:
    def test_generate_uses_template_when_no_client(self):
        service = RAGService(api_key=None)
        response = service.generate(_q(), [_rd()])
        assert response.used_llm is False
        assert response.model_name == "template_fallback"
        assert response.answer  # template produces non-empty output

    def test_generate_with_no_profile_defaults_patient(self):
        service = RAGService(api_key=None)
        response = service.generate(_q(profile=None), [_rd()])
        assert response.profile_type == UserProfileType.PATIENT

    def test_generate_caps_sources_at_max_context_docs(self):
        service = RAGService(api_key=None, max_context_docs=2)
        docs = [_rd(f"d{i}") for i in range(5)]
        response = service.generate(_q(), docs)
        assert len(response.sources) <= 2

    def test_generate_override_max_context_docs(self):
        service = RAGService(api_key=None, max_context_docs=2)
        docs = [_rd(f"d{i}") for i in range(5)]
        response = service.generate(_q(), docs, max_context_docs=1)
        assert len(response.sources) <= 1


# ---------------------------------------------------------------------------
# Ranker integration
# ---------------------------------------------------------------------------


class TestRankerIntegration:
    def test_ranker_called_when_provided(self):
        class _CapturingRanker(BaseRanker):
            def __init__(self):
                self.called_with: list[RetrievedDocument] | None = None

            def rerank(self, query, retrieved):
                self.called_with = list(retrieved)
                # Reverse the input list so we can assert the ranker actually ran
                return list(reversed(retrieved))

        ranker = _CapturingRanker()
        service = RAGService(api_key=None, ranker=ranker, max_context_docs=10)
        docs = [_rd("a"), _rd("b"), _rd("c")]
        response = service.generate(_q(), docs)

        assert ranker.called_with is not None
        # Sources should reflect the reranker's output order
        assert [r.document.doc_id for r in response.sources] == ["c", "b", "a"]

    def test_no_ranker_preserves_original_order(self):
        service = RAGService(api_key=None, max_context_docs=10)
        docs = [_rd("a"), _rd("b"), _rd("c")]
        response = service.generate(_q(), docs)
        assert [r.document.doc_id for r in response.sources] == ["a", "b", "c"]
