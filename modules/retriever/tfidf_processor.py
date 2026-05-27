"""TF-IDF processor - builds sparse matrix from IndexedCorpus.

Receives IndexedCorpus for both documents and queries.
Filters query terms that are not in the document vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from sklearn.preprocessing import normalize

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
        n_terms = len(self._vocabulary)

        # IDF (smoothed): log((N+1) / (df+1)) + 1, vectorized over the vocabulary.
        # df is the document frequency = number of postings for the term.
        df = np.fromiter(
            (len(corpus.inverted_index.get(term, [])) for term in self._vocabulary),
            dtype=np.float32,
            count=n_terms,
        )
        self._idf = (np.log((self._n_docs + 1) / (df + 1)) + 1.0).astype(np.float32)

        # Build the sparse TF-IDF matrix. Gather the (doc, term, tf) triples per
        # term as arrays, then apply the TF-IDF weight (log1p(tf) * idf) in a
        # single vectorized pass — far cheaper than a Python loop over every
        # posting. The result is identical to the per-element computation.
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        tf_parts: list[np.ndarray] = []
        for term, postings in corpus.inverted_index.items():
            term_idx = self._term_to_idx.get(term)
            if term_idx is None or not postings:
                continue
            arr = np.asarray(postings, dtype=np.int64)  # shape (n_postings, 2)
            row_parts.append(arr[:, 0])
            col_parts.append(np.full(len(postings), term_idx, dtype=np.int64))
            tf_parts.append(arr[:, 1])

        if row_parts:
            rows = np.concatenate(row_parts)
            cols = np.concatenate(col_parts)
            tfs = np.concatenate(tf_parts).astype(np.float32)
            data = np.log1p(tfs) * self._idf[cols]
        else:
            rows = cols = data = np.empty(0)

        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self._n_docs, n_terms),
            dtype=np.float32,
        )
        return normalize(matrix, norm="l2", axis=1)

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
                data.append(np.log1p(tf) * self._idf[term_idx])

        matrix = csr_matrix(
            (data, ([0] * len(data), indices)),
            shape=(1, len(self._vocabulary)),
            dtype=np.float32,
        )
        return normalize(matrix, norm="l2", axis=1)

    def transform_corpus(self, corpus: IndexedCorpus) -> spmatrix:
        """Transform a multi-document corpus using the fitted vocabulary and IDF.

        Generalizes :meth:`transform` (single query) to N documents. Used for
        dynamic indexing (folding-in): new documents are weighted with the
        *existing* IDF and projected onto the *existing* vocabulary, so terms
        absent from the fitted vocabulary are dropped. The matrix it returns is
        fed to :meth:`LSIModel.project_documents`.

        Args:
            corpus: IndexedCorpus of the new documents. Only its
                ``inverted_index`` (term → [(doc_idx, tf), ...]) and document
                count are read; the corpus's own vocabulary is ignored in favor
                of the fitted one.

        Returns:
            Sparse TF-IDF matrix of shape ``(n_new_docs, n_terms)`` aligned to
            the fitted vocabulary.

        Raises:
            RuntimeError: If the processor has not been fitted.
        """
        if self._term_to_idx is None or self._idf is None:
            raise RuntimeError("Must call fit() before transform_corpus()")

        n_docs = len(corpus.documents)
        n_terms = len(self._vocabulary)

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for term, postings in corpus.inverted_index.items():
            term_idx = self._term_to_idx.get(term)
            if term_idx is None:
                continue  # out-of-vocabulary term — dropped by folding-in
            weight = self._idf[term_idx]
            for doc_idx, tf in postings:
                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(np.log1p(tf) * weight)

        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_docs, n_terms),
            dtype=np.float32,
        )
        return normalize(matrix, norm="l2", axis=1)

    @property
    def vocabulary(self) -> list[str]:
        """Return fitted vocabulary."""
        if self._vocabulary is None:
            raise RuntimeError("Must call fit() first")
        return self._vocabulary

    @property
    def n_docs(self) -> int:
        """Number of documents the processor was fitted on."""
        return self._n_docs

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
