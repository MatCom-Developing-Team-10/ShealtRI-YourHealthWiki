"""Tests for LSIModel — TruncatedSVD over a TF-IDF matrix."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from modules.retriever.lsi_model import LSIModel


def _matrix(rows: int, cols: int, seed: int = 0):
    """Generate a deterministic non-empty sparse matrix for SVD."""
    rng = np.random.default_rng(seed)
    dense = rng.random((rows, cols)).astype(np.float32)
    # Ensure no all-zero rows so SVD has real signal
    return csr_matrix(dense)


class TestFit:
    def test_fit_returns_doc_vectors_with_expected_shape(self):
        m = _matrix(rows=10, cols=20)
        model = LSIModel(n_components=5)
        vectors = model.fit(m)
        assert len(vectors) == 10
        assert all(len(v) == 5 for v in vectors)

    def test_is_fitted_flag(self):
        model = LSIModel(n_components=3)
        assert model.is_fitted is False
        model.fit(_matrix(5, 8))
        assert model.is_fitted is True

    def test_n_components_capped_when_corpus_too_small(self):
        # n_terms=3, n_docs=2 → effective_k = min(10, 2, 1) = 1
        m = _matrix(rows=2, cols=3)
        model = LSIModel(n_components=10)
        vectors = model.fit(m)
        assert model.n_components == 1
        assert all(len(v) == 1 for v in vectors)

    def test_single_document_corpus_raises_valueerror(self):
        """Single document corpus should raise ValueError with clear message.

        LSI requires at least 2 documents to compute SVD. A single document
        cannot be decomposed into latent dimensions.
        """
        m = _matrix(rows=1, cols=5)
        model = LSIModel(n_components=2)
        with pytest.raises(ValueError, match="at least 2 documents"):
            model.fit(m)


class TestProjectQuery:
    def test_project_before_fit_raises(self):
        model = LSIModel()
        q = csr_matrix((1, 5), dtype=np.float32)
        with pytest.raises(RuntimeError):
            model.project_query(q)

    def test_project_after_fit_returns_vector_of_correct_length(self):
        m = _matrix(rows=10, cols=20)
        model = LSIModel(n_components=4)
        model.fit(m)
        q = _matrix(rows=1, cols=20, seed=99)
        vec = model.project_query(q)
        assert isinstance(vec, list)
        assert len(vec) == 4

    def test_identical_query_to_doc_yields_high_similarity(self):
        # Construct a small deterministic matrix and a query that mirrors doc 0
        m = _matrix(rows=5, cols=8)
        model = LSIModel(n_components=3)
        doc_vectors = model.fit(m)
        # Use exactly the row of doc 0 as the query
        q = csr_matrix(m.toarray()[0:1])
        q_latent = model.project_query(q)
        # q_latent should be very close to doc_vectors[0]
        np.testing.assert_allclose(np.array(q_latent), np.array(doc_vectors[0]), rtol=1e-5)


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        m = _matrix(rows=6, cols=10)
        model = LSIModel(n_components=3)
        original = model.fit(m)
        model.save(tmp_path / "lsi")

        loaded = LSIModel.load(tmp_path / "lsi")
        assert loaded.is_fitted is True
        # Re-projecting the same matrix yields the same vectors as originally fitted
        q = csr_matrix(m.toarray()[0:1])
        loaded_q = loaded.project_query(q)
        original_q = model.project_query(q)
        np.testing.assert_allclose(loaded_q, original_q, rtol=1e-5)
