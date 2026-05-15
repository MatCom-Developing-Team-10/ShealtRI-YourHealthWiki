"""Tests for TfidfProcessor — TF-IDF matrix construction from IndexedCorpus."""

from __future__ import annotations

import numpy as np
import pytest

from core.interfaces import IndexedCorpus
from core.models import Document
from modules.retriever.tfidf_processor import TfidfProcessor


def _corpus(
    docs: list[tuple[str, str]],
    inverted_index: dict[str, list[tuple[int, int]]],
    vocabulary: list[str],
) -> IndexedCorpus:
    return IndexedCorpus(
        documents=[Document(d_id, "", "") for d_id, _ in docs],
        processed_texts=[t for _, t in docs],
        inverted_index=inverted_index,
        vocabulary=vocabulary,
    )


class TestFit:
    def test_returns_sparse_matrix_with_correct_shape(self):
        c = _corpus(
            [("d1", "alpha beta"), ("d2", "beta gamma")],
            {"alpha": [(0, 1)], "beta": [(0, 1), (1, 1)], "gamma": [(1, 1)]},
            ["alpha", "beta", "gamma"],
        )
        tfidf = TfidfProcessor()
        matrix = tfidf.fit(c)
        assert matrix.shape == (2, 3)

    def test_idf_higher_for_rarer_terms(self):
        c = _corpus(
            [("d1", "common rare1"), ("d2", "common rare2"), ("d3", "common")],
            {
                "common": [(0, 1), (1, 1), (2, 1)],
                "rare1": [(0, 1)],
                "rare2": [(1, 1)],
            },
            ["common", "rare1", "rare2"],
        )
        tfidf = TfidfProcessor()
        tfidf.fit(c)
        idf = tfidf._idf
        # 'common' is in 3/3 docs → low IDF; 'rare1' is in 1/3 docs → high IDF
        common_idx = tfidf._term_to_idx["common"]
        rare_idx = tfidf._term_to_idx["rare1"]
        assert idf[rare_idx] > idf[common_idx]

    def test_vocabulary_property(self):
        c = _corpus(
            [("d1", "x")], {"x": [(0, 1)]}, ["x"]
        )
        tfidf = TfidfProcessor()
        tfidf.fit(c)
        assert tfidf.vocabulary == ["x"]

    def test_vocabulary_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _ = TfidfProcessor().vocabulary


class TestTransform:
    def test_transform_before_fit_raises(self):
        tfidf = TfidfProcessor()
        c = _corpus([("q", "x")], {"x": [(0, 1)]}, ["x"])
        with pytest.raises(RuntimeError):
            tfidf.transform(c)

    def test_query_terms_in_vocabulary_pass_through(self):
        c = _corpus(
            [("d1", "alpha beta"), ("d2", "beta gamma")],
            {"alpha": [(0, 1)], "beta": [(0, 1), (1, 1)], "gamma": [(1, 1)]},
            ["alpha", "beta", "gamma"],
        )
        tfidf = TfidfProcessor()
        tfidf.fit(c)

        query = _corpus([("q", "alpha")], {"alpha": [(0, 1)]}, ["alpha"])
        vec = tfidf.transform(query)
        assert vec.shape == (1, 3)
        # Only the 'alpha' column should be non-zero
        dense = vec.toarray()[0]
        alpha_idx = tfidf._term_to_idx["alpha"]
        assert dense[alpha_idx] > 0
        for i, val in enumerate(dense):
            if i != alpha_idx:
                assert val == 0

    def test_oov_query_terms_filtered(self):
        c = _corpus(
            [("d1", "alpha")], {"alpha": [(0, 1)]}, ["alpha"]
        )
        tfidf = TfidfProcessor()
        tfidf.fit(c)
        query = _corpus(
            [("q", "unknown unseen")],
            {"unknown": [(0, 1)], "unseen": [(0, 1)]},
            ["unknown", "unseen"],
        )
        vec = tfidf.transform(query)
        assert vec.nnz == 0  # no non-zero entries


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        c = _corpus(
            [("d1", "alpha beta"), ("d2", "beta gamma")],
            {"alpha": [(0, 1)], "beta": [(0, 1), (1, 1)], "gamma": [(1, 1)]},
            ["alpha", "beta", "gamma"],
        )
        tfidf = TfidfProcessor()
        tfidf.fit(c)
        tfidf.save(tmp_path / "models")

        restored = TfidfProcessor.load(tmp_path / "models")
        assert restored.vocabulary == tfidf.vocabulary
        np.testing.assert_array_equal(restored._idf, tfidf._idf)
        assert restored._n_docs == tfidf._n_docs
