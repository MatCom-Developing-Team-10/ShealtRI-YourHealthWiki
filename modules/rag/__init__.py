"""RAG module — profile-aware answer generation over retrieved documents."""

from .service import RAGService
from .user_profiles import ProfileRegistry

__all__ = ["RAGService", "ProfileRegistry"]
