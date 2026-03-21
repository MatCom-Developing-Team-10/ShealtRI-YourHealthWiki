"""LSI retriever service that applies query correction before retrieval."""

from __future__ import annotations

import re

from core.interfaces import BaseRetriever
from core.models import Document, Query, RetrievedDocument

from .lsi_model import LSIModel
from .spell_checker import TrieSpellChecker


class LSIRetriever(BaseRetriever):
    """Retriever based on TF-IDF + TruncatedSVD (LSI)."""

    def __init__(self, n_components: int = 100, max_spell_distance: int = 2) -> None:
        raise NotImplementedError

    def fit(self, documents: list[Document]) -> None:
        """Fit internal LSI model and initialize spell checker vocabulary."""
        raise NotImplementedError

    def _normalize_query(self, text: str) -> str:
        raise NotImplementedError

    def retrieve(self, query: Query, top_k: int = 10) -> list[RetrievedDocument]:
        """Retrieve the top_k most relevant documents for a query."""
        raise NotImplementedError

    def save(self, model_dir: str) -> None:
        """Persist the fitted model artifacts."""
        raise NotImplementedError

    @classmethod
    def load(cls, model_dir: str, max_spell_distance: int = 2) -> "LSIRetriever":
        """Restore a retriever and spell checker from persisted artifacts."""
        raise NotImplementedError
