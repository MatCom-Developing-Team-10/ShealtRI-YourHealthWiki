"""TF-IDF processor that consumes precomputed data from the indexer.

This module sits between the indexer and the LSI model:
    IndexedCorpus (adapter) → TfidfProcessor → sparse matrix → LSIModel
"""

from __future__ import annotations

from pathlib import Path

import joblib
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from core.interfaces import IndexedCorpus


class TfidfProcessor:
    """Builds and holds a TF-IDF matrix from an IndexedCorpus.

    The processor respects the vocabulary provided by the indexer
    (via ``IndexedCorpus.vocabulary``).  If the indexer returns
    ``None``, the vectorizer discovers the vocabulary automatically.

    Typical usage::

        corpus: IndexedCorpus = indexer.build()   # future
        tfidf = TfidfProcessor()
        matrix = tfidf.fit(corpus)                # sparse (docs × terms)
        query_vec = tfidf.transform("headache")   # sparse (1 × terms)
    """

    def __init__(
        self,
        max_features: int | None = 10_000,
        min_df: int = 1,
        max_df: float = 0.95,
    ) -> None:
        """Initialize vectorizer hyper-parameters.

        Args:
            max_features: Cap on vocabulary size.  Ignored when the corpus
                provides an explicit vocabulary.
            min_df: Minimum document frequency for a term.
            max_df: Maximum document frequency (as fraction).
        """
        self._max_features = max_features
        self._min_df = min_df
        self._max_df = max_df

        self._vectorizer: TfidfVectorizer | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, corpus: IndexedCorpus) -> spmatrix:
        """Build the TF-IDF matrix from the indexed corpus.

        If the corpus supplies an explicit vocabulary the vectorizer will
        use it (and ``max_features`` / ``min_df`` / ``max_df`` are
        effectively overridden).  Otherwise the vectorizer discovers the
        vocabulary from the preprocessed texts.

        Args:
            corpus: An object implementing ``IndexedCorpus``.

        Returns:
            Sparse TF-IDF matrix of shape ``(n_documents, n_terms)``.
        """
        vocab = corpus.vocabulary

        if vocab is not None:
            # Indexer provides an explicit vocabulary — honour it.
            self._vectorizer = TfidfVectorizer(vocabulary=vocab)
        else:
            self._vectorizer = TfidfVectorizer(
                max_features=self._max_features,
                min_df=self._min_df,
                max_df=self._max_df,
            )

        tfidf_matrix: spmatrix = self._vectorizer.fit_transform(
            corpus.processed_texts,
        )
        self._fitted = True
        return tfidf_matrix

    def transform(self, text: str) -> spmatrix:
        """Project a single text into the fitted TF-IDF space.

        Args:
            text: Raw or preprocessed query text.

        Returns:
            Sparse TF-IDF vector of shape ``(1, n_terms)``.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._fitted or self._vectorizer is None:
            raise RuntimeError("TfidfProcessor must be fitted before transform.")
        return self._vectorizer.transform([text])

    @property
    def vocabulary(self) -> list[str]:
        """Return the fitted vocabulary as an ordered list of terms.

        Useful for initializing the spell-checker with corpus terms.
        """
        if not self._fitted or self._vectorizer is None:
            raise RuntimeError("TfidfProcessor must be fitted first.")
        return list(self._vectorizer.get_feature_names_out())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted vectorizer to disk.

        Args:
            path: Directory where ``tfidf.joblib`` will be written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, path / "tfidf.joblib")

    @classmethod
    def load(cls, path: str | Path) -> "TfidfProcessor":
        """Restore a fitted TfidfProcessor from disk.

        Args:
            path: Directory containing ``tfidf.joblib``.

        Returns:
            A ready-to-use TfidfProcessor instance.
        """
        path = Path(path)
        instance = cls()
        instance._vectorizer = joblib.load(path / "tfidf.joblib")
        instance._fitted = True
        return instance
