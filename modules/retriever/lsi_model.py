"""Core LSI model implementation with persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.models import Document


class LSIModel:
    """Encapsulates TF-IDF + TruncatedSVD for latent semantic retrieval."""

    def __init__(
        self,
        n_components: int = 100,
        max_features: int = 10_000,
        min_df: int = 1,
        max_df: float = 0.95,
        random_state: int = 42,
    ) -> None:
        raise NotImplementedError

    def fit(self, documents: Sequence[Document]) -> None:
        """Train TF-IDF and SVD from the provided corpus."""
        raise NotImplementedError

    def project_query(self, query_text: str) -> np.ndarray:
        """Project a query to the same latent space used by document vectors."""
        raise NotImplementedError

    def retrieve(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """Return document indices and scores ordered by cosine similarity."""
        raise NotImplementedError

    def save(self, model_dir: str | Path) -> None:
        """Persist the fitted components to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, model_dir: str | Path) -> "LSIModel":
        """Load a fitted model from disk."""
        raise NotImplementedError
