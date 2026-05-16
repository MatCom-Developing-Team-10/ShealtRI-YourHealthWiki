"""Shared domain models for the SRI pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.interfaces import IndexedCorpus


class UserProfileType(Enum):
    """Supported user profiles for profile-aware RAG responses."""

    PATIENT = "paciente"
    MEDICAL_STUDENT = "estudiante_medicina"
    MEDICAL_PROFESSIONAL = "profesional_medico"
    DIAGNOSTIC_ASSISTANT = "diagnostico_asistido"
    NATURAL_MEDICINE = "medicina_natural"
    CAREGIVER = "cuidador_familiar"


@dataclass(slots=True)
class UserProfile:
    """Captures the user's role and communication preferences.

    Attributes:
        profile_type: Enum identifying the role.
        name: Human-readable label shown in UI (Spanish).
        custom_instructions: Optional free-text override appended to the
            system prompt for per-user fine-tuning.
    """

    profile_type: UserProfileType
    name: str
    custom_instructions: str = ""


@dataclass(slots=True)
class Query:
    """Represents an incoming user query.

    Attributes:
        text: Raw query string entered by the user.
        indexed_corpus: Preprocessed query as IndexedCorpus for retrieval.
            Built by the pipeline before calling the retriever.
        metadata: Optional metadata associated with the query.
        user_profile: User profile for profile-aware RAG responses.
    """

    text: str
    indexed_corpus: IndexedCorpus | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    user_profile: UserProfile | None = None


@dataclass(slots=True)
class Document:
    """Represents a retrievable document in the corpus.

    Attributes:
        doc_id: Unique identifier for a document.
        text: Plain text content used by retrievers.
        metadata: Optional metadata (source URL, title, tags, etc.).
    """

    doc_id: str
    text: str
    url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedDocument:
    """Represents a retrieved document and its relevance score."""

    document: Document
    score: float


@dataclass(slots=True)
class RAGResponse:
    """Encapsulates a generated answer and its provenance.

    Attributes:
        answer: Generated text in Spanish, adapted to the user profile.
        profile_type: The profile that shaped the generation.
        sources: Documents used as context for generation.
        used_llm: True if an LLM produced the answer; False if template fallback.
        model_name: LLM model identifier (e.g. "llama-3.1-8b-instant"), or
            "template_fallback" when the LLM was unavailable.
        query_text: Original query text — stored for logging/display.
    """

    answer: str
    profile_type: UserProfileType
    sources: list[RetrievedDocument]
    used_llm: bool
    model_name: str
    query_text: str
