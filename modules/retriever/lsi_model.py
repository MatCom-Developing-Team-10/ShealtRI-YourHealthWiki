"""Core LSI model — dimensionality reduction over a precomputed TF-IDF matrix.

This module is intentionally decoupled from TF-IDF construction.
It receives a sparse TF-IDF matrix and applies TruncatedSVD.

    TfidfProcessor.fit(corpus)  →  sparse matrix  →  LSIModel.fit(matrix)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, spmatrix, vstack
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
        # n_iter=7 balances randomized-SVD accuracy and speed: for TF-IDF
        # matrices the spectrum decays quickly, so 7 power iterations match the
        # quality of higher counts while running noticeably faster.
        self._n_iter = 7
        self._svd = TruncatedSVD(
            n_components=n_components,
            algorithm="randomized",
            n_iter=self._n_iter,
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
            document, each of length ``n_components`` (or fewer if the
            corpus is too small for the requested ``n_components``).

        Raises:
            ValueError: If the corpus is empty.

        Notes:
            scikit-learn's ``TruncatedSVD`` requires
            ``n_components < min(n_samples, n_features)``. For a
            single-document corpus this constraint reduces ``n_components``
            below 1 and the decomposition is rejected. We handle that
            degenerate case by padding the matrix with a synthetic zero
            row so SVD has two samples to work with, then discard the
            padding's latent vector. The fitted ``_svd`` remains usable for
            ``project_query`` / ``project_documents`` against the original
            (single) document, which is what callers expect.
        """
        n_docs, n_terms = tfidf_matrix.shape

        if n_docs == 0:
            raise ValueError(
                "LSI requires at least one document to fit; got an empty corpus."
            )

        # Padded SVD path for the degenerate single-document case.
        if n_docs == 1:
            zero_row = csr_matrix((1, n_terms), dtype=tfidf_matrix.dtype)
            matrix_for_svd: spmatrix = vstack([tfidf_matrix, zero_row])
            keep_rows = 1
        else:
            matrix_for_svd = tfidf_matrix
            keep_rows = n_docs

        n_samples = matrix_for_svd.shape[0]

        # SVD requires k < min(n_samples, n_features); clamp to [1, ...].
        effective_k = max(1, min(self.n_components, n_terms - 1, n_samples - 1))
        if effective_k < self.n_components:
            self._svd = TruncatedSVD(
                n_components=effective_k,
                algorithm="randomized",
                n_iter=self._n_iter,
                random_state=self._svd.random_state,
            )
            self.n_components = effective_k

        doc_vectors: np.ndarray = self._svd.fit_transform(matrix_for_svd)
        self._fitted = True

        # Drop the padding row's vector when the single-doc path was used.
        return doc_vectors[:keep_rows].tolist()

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

    def project_documents(self, tfidf_matrix: spmatrix) -> list[list[float]]:
        """Project new documents into the existing latent space (folding-in).

        This is the dynamic-indexing primitive: it maps new documents onto the
        already-trained SVD basis *without* re-fitting the model, exactly as the
        lecture (Conf_2) describes for incremental updates ("añadir nuevos
        documentos sin reindexar todo"). Mathematically it applies the folding-in
        formula ``d_proj = d · Vk`` (i.e. ``TruncatedSVD.transform``, which
        computes ``X · components_.T``), to every row of the matrix.

        Because the SVD basis stays fixed, terms absent from the original
        vocabulary contribute nothing — that information loss is why a periodic
        full ``fit`` (the "balanceo" step) is still needed once enough new
        documents accumulate.

        Args:
            tfidf_matrix: Sparse matrix of shape ``(n_new_docs, n_terms)`` built
                with the *same* vocabulary used at fit time
                (``TfidfProcessor.transform_corpus``).

        Returns:
            Latent vectors — one list of length ``n_components`` per document.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("LSIModel must be fitted before projecting documents.")
        latent: np.ndarray = self._svd.transform(tfidf_matrix)
        return latent.tolist()

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
