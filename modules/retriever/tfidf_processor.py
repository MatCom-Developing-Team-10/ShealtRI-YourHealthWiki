<<<<<<< HEAD
"""TF-IDF processor - builds sparse matrix from IndexedCorpus.

Receives IndexedCorpus for both documents and queries.
Filters query terms that are not in the document vocabulary.
=======
"""TF-IDF processor that consumes precomputed data from the indexer.

This module sits between the indexer and the LSI model:
    IndexedCorpus (adapter) → TfidfProcessor → sparse matrix → LSIModel
>>>>>>> 2491ed1 (feat: Enhance LSI retrieval system with new data structures and storage layers)
"""

from __future__ import annotations

from pathlib import Path

import joblib
<<<<<<< HEAD
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
=======
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
>>>>>>> 2491ed1 (feat: Enhance LSI retrieval system with new data structures and storage layers)

from core.interfaces import IndexedCorpus


class TfidfProcessor:
<<<<<<< HEAD
    """Builds TF-IDF matrix from IndexedCorpus.

    Both documents and queries come as IndexedCorpus.
    Query terms not in the document vocabulary are filtered out.
    """

    def __init__(self) -> None:
        self._vocabulary: list[str] | None = None
        self._term_to_idx: dict[str, int] | None = None
        self._idf: np.ndarray | None = None
        self._n_docs: int = 0

    def fit(self, corpus: IndexedCorpus) -> spmatrix:
        """Build TF-IDF matrix from document corpus.

        Args:
            corpus: IndexedCorpus with documents, inverted_index, vocabulary.

        Returns:
            Sparse TF-IDF matrix (n_docs × n_terms).
        """
        self._n_docs = len(corpus.documents)
        self._vocabulary = corpus.vocabulary
        self._term_to_idx = {term: idx for idx, term in enumerate(self._vocabulary)}

        # Compute IDF: log((N+1) / (df+1)) + 1
        self._idf = np.zeros(len(self._vocabulary), dtype=np.float32)
        for term_idx, term in enumerate(self._vocabulary):
            df = len(corpus.inverted_index.get(term, []))
            self._idf[term_idx] = np.log((self._n_docs + 1) / (df + 1)) + 1.0

        # Build sparse matrix: TF-IDF = TF * IDF
        rows, cols, data = [], [], []
        for term, postings in corpus.inverted_index.items():
            if term not in self._term_to_idx:
                continue
            term_idx = self._term_to_idx[term]
            term_idf = self._idf[term_idx]
            for doc_idx, tf in postings:
                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(tf * term_idf)

        return csr_matrix(
            (data, (rows, cols)),
            shape=(self._n_docs, len(self._vocabulary)),
            dtype=np.float32,
        )

    def transform(self, query_corpus: IndexedCorpus) -> spmatrix:
        """Transform query corpus to TF-IDF vector.

        Args:
            query_corpus: IndexedCorpus with 1 document (the query).
                Terms not in the document vocabulary are filtered out.

        Returns:
            Sparse TF-IDF vector (1 × n_terms).
        """
        if self._term_to_idx is None or self._idf is None:
            raise RuntimeError("Must call fit() before transform()")

        # Query comes as IndexedCorpus with 1 document
        # Extract term frequencies from inverted_index
        indices, data = [], []

        for term, postings in query_corpus.inverted_index.items():
            # Filter: only terms in document vocabulary
            if term not in self._term_to_idx:
                continue

            # Get term frequency for the query (first posting, first doc)
            if postings:
                tf = postings[0][1]  # (doc_idx, freq) -> freq
                term_idx = self._term_to_idx[term]
                indices.append(term_idx)
                data.append(tf * self._idf[term_idx])

        return csr_matrix(
            (data, ([0] * len(data), indices)),
            shape=(1, len(self._vocabulary)),
            dtype=np.float32,
        )

    @property
    def vocabulary(self) -> list[str]:
        """Return fitted vocabulary."""
        if self._vocabulary is None:
            raise RuntimeError("Must call fit() first")
        return self._vocabulary

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vocabulary": self._vocabulary, "idf": self._idf, "n_docs": self._n_docs},
            Path(path) / "tfidf.joblib",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TfidfProcessor":
        """Load model from disk."""
        data = joblib.load(Path(path) / "tfidf.joblib")
        instance = cls()
        instance._vocabulary = data["vocabulary"]
        instance._idf = data["idf"]
        instance._n_docs = data["n_docs"]
        instance._term_to_idx = {t: i for i, t in enumerate(instance._vocabulary)}
=======
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
>>>>>>> 2491ed1 (feat: Enhance LSI retrieval system with new data structures and storage layers)
        return instance
