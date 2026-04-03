"""TF-IDF processor - builds sparse matrix from IndexedCorpus.

Receives IndexedCorpus for both documents and queries.
Filters query terms that are not in the document vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, spmatrix

from core.interfaces import IndexedCorpus


class TfidfProcessor:
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
        return instance
