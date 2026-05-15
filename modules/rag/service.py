"""RAGService — profile-aware answer generation using Groq API.

Architecture:
    1. Read user profile from query.user_profile (fall back to PATIENT).
    2. Build context block from retrieved documents.
    3. Render prompt using the profile's system_prompt template.
    4. Call Groq API to generate an answer.
    5. On any API failure, fall back to template-based response.
    6. Return RAGResponse with provenance metadata.

Groq is called via the groq SDK with OpenAI-compatible chat.completions interface.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.interfaces import BaseRAG
from core.models import (
    Query,
    RAGResponse,
    RetrievedDocument,
    UserProfile,
    UserProfileType,
)
from .prompt_templates import (
    build_context_block,
    build_fallback_response,
    render_prompt,
)
from .user_profiles import ProfileRegistry

logger = logging.getLogger(__name__)

_FALLBACK_MODEL_NAME = "template_fallback"


class RAGService(BaseRAG):
    """Profile-aware RAG using Groq API as the LLM backend.

    Attributes:
        api_key: Groq API key from environment or constructor.
        model_name: Groq model to use (e.g. "llama-3.1-8b-instant").
        max_context_docs: Max retrieved documents included in the prompt.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        max_context_docs: int = 3,
    ) -> None:
        """Initialize the RAG service.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY
                environment variable. If both are None, fallback mode is
                enabled (only template responses).
            model_name: Groq model identifier. If None, reads from
                GROQ_MODEL environment variable. Defaults to
                "llama-3.1-8b-instant".
            max_context_docs: Number of retrieved documents to include
                in the LLM context window.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = (
            model_name
            or os.environ.get("GROQ_MODEL")
            or "llama-3.1-8b-instant"
        )
        self.max_context_docs = max_context_docs
        self._client = None

        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info("Groq API initialized (model: %s)", self.model_name)
            except ImportError:
                logger.warning(
                    "groq package not installed. "
                    "RAG will use template fallback only."
                )
            except Exception as exc:
                logger.warning("Failed to initialize Groq API: %s", exc)

    # ------------------------------------------------------------------
    # Public API — BaseRAG contract
    # ------------------------------------------------------------------

    def generate(
        self,
        query: Query,
        retrieved_docs: list[RetrievedDocument],
        max_context_docs: int | None = None,
    ) -> RAGResponse:
        """Generate a profile-adapted answer.

        Args:
            query: Original query; query.user_profile carries the profile.
            retrieved_docs: Ranked documents from LSIRetriever.
            max_context_docs: Override for the instance default.

        Returns:
            RAGResponse with answer, profile metadata, and provenance.
        """
        n_docs = max_context_docs if max_context_docs is not None else self.max_context_docs
        profile = self._resolve_profile(query.user_profile)
        profile_config = ProfileRegistry.get(profile.profile_type)

        context_block = build_context_block(retrieved_docs, max_docs=n_docs)
        prompt = render_prompt(profile_config, query.text, context_block)

        answer, used_llm, model_used = self._call_llm(prompt)

        if not used_llm:
            answer = build_fallback_response(
                profile_config, query.text, retrieved_docs, max_docs=n_docs
            )

        return RAGResponse(
            answer=answer,
            profile_type=profile.profile_type,
            sources=retrieved_docs[:n_docs],
            used_llm=used_llm,
            model_name=model_used,
            query_text=query.text,
        )

    # ------------------------------------------------------------------
    # Groq integration
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> tuple[str, bool, str]:
        """Call the Groq API to generate a response.

        Args:
            prompt: Complete rendered prompt string.

        Returns:
            Tuple of (answer_text, used_llm, model_name).
            If Groq is unavailable, used_llm=False and answer_text is "".
        """
        if not self._client:
            logger.debug("Groq API not available; using template fallback")
            return "", False, _FALLBACK_MODEL_NAME

        try:
            logger.debug("Calling Groq API with model %s", self.model_name)

            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=512,
            )
            answer = response.choices[0].message.content or ""
            answer = answer.strip()

            logger.info("Groq response received (%d chars)", len(answer))
            if not answer:
                logger.warning("Groq returned an empty response")
                return "", False, _FALLBACK_MODEL_NAME
            return answer, True, self.model_name

        except Exception as exc:
            logger.error("Groq API call failed: %s — using template fallback", exc)
            return "", False, _FALLBACK_MODEL_NAME

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if Groq API is configured and ready.

        Returns:
            True if API key is set and groq client is available.
        """
        return bool(self._client and self.api_key)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_profile(user_profile: UserProfile | None) -> UserProfile:
        """Return the user profile to use, defaulting to PATIENT.

        Args:
            user_profile: Profile from the query, or None.

        Returns:
            A guaranteed non-None UserProfile.
        """
        if user_profile is not None:
            return user_profile
        return UserProfile(
            profile_type=UserProfileType.PATIENT,
            name="Paciente",
        )
