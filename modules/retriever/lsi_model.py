"""Core LSI model — dimensionality reduction over a precomputed TF-IDF matrix.

This module is intentionally decoupled from TF-IDF construction.
It receives a sparse TF-IDF matrix and applies TruncatedSVD.

    TfidfProcessor.fit(corpus)  →  sparse matrix  →  LSIModel.fit(matrix)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class LSIModel:
    """Applies TruncatedSVD to a precomputed TF-IDF matrix.

    Responsibilities:
        - Fit: receive sparse TF-IDF matrix → produce document vectors in
          latent space.
        - Project: receive a TF-IDF query vector → project into the same
          latent space.
        - Persist / load the SVD model to/from disk.
    """

    def __init__(
        self,
        n_components: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialize SVD hyper-parameters.

        Args:
            n_components: Number of latent dimensions (k).
            random_state: Seed for reproducibility.
        """
        self.n_components = n_components
        self._svd = TruncatedSVD(
            n_components=n_components,
            random_state=random_state,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, tfidf_matrix: spmatrix) -> list[list[float]]:
        """Reduce dimensionality and return document vectors.

        Args:
            tfidf_matrix: Sparse matrix of shape ``(n_docs, n_terms)``
                produced by ``TfidfProcessor.fit()``.

        Returns:
            Document vectors in latent space — list of lists, one per
            document, each of length ``n_components``.
        """
        n_docs, n_terms = tfidf_matrix.shape
        effective_k = min(self.n_components, n_terms - 1, n_docs - 1)
        if effective_k < self.n_components:
            self._svd = TruncatedSVD(
                n_components=effective_k,
                random_state=self._svd.random_state,
            )
            self.n_components = effective_k

        doc_vectors: np.ndarray = self._svd.fit_transform(tfidf_matrix)
        self._fitted = True
        return doc_vectors.tolist()

    def project_query(self, query_tfidf: spmatrix) -> list[float]:
        """Project a TF-IDF query vector into the latent space.

        Args:
            query_tfidf: Sparse vector of shape ``(1, n_terms)`` produced by
                ``TfidfProcessor.transform(text)``.

        Returns:
            Query vector in latent space (length ``n_components``).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("LSIModel must be fitted before projecting queries.")
        query_latent: np.ndarray = self._svd.transform(query_tfidf)
        return query_latent[0].tolist()

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted SVD model to disk.

        Args:
            path: Directory where ``svd.joblib`` will be written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._svd, path / "svd.joblib")

    @classmethod
    def load(cls, path: str | Path) -> "LSIModel":
        """Restore a fitted LSIModel from disk.

        Args:
            path: Directory containing ``svd.joblib``.

        Returns:
            A ready-to-use LSIModel instance.
        """
        path = Path(path)
        instance = cls()
        instance._svd = joblib.load(path / "svd.joblib")
        instance._fitted = True
        instance.n_components = instance._svd.n_components
        return instance
