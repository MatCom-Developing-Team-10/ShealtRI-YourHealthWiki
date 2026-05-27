"""Tests for dynamic indexing (Conf_2) — LSI folding-in + rebalance tracking.

These verify the incremental-update path added to the retriever:
    - add_documents() folds new docs into the existing latent space
    - the SVD model is NOT re-fitted (folding-in, not rebuild)
    - a folded-in document becomes retrievable without a full reindex
    - incremental_fraction / needs_rebalance track when a refit is warranted

The helpers build an IndexedCorpus directly (no spaCy) so the suite stays fast.
"""

from __future__ import annotations

from collections import Counter

import pytest

from core.interfaces import IndexedCorpus
from core.models import Document, Query
from modules.retriever.service import LSIRetriever


def _build_indexed_corpus(rows: list[tuple[str, str, list[str]]]) -> IndexedCorpus:
    """Build an IndexedCorpus from rows of (doc_id, url, tokens)."""
    documents = [Document(doc_id, " ".join(tokens), url) for doc_id, url, tokens in rows]
    processed_texts = [" ".join(tokens) for _, _, tokens in rows]
    vocabulary_set: set[str] = set()
    for _, _, tokens in rows:
        vocabulary_set.update(tokens)
    vocabulary = sorted(vocabulary_set)
    inverted_index: dict[str, list[tuple[int, int]]] = {}
    for doc_idx, (_, _, tokens) in enumerate(rows):
        for term, tf in Counter(tokens).items():
            inverted_index.setdefault(term, []).append((doc_idx, tf))
    return IndexedCorpus(
        documents=documents,
        processed_texts=processed_texts,
        inverted_index=inverted_index,
        vocabulary=vocabulary,
    )


def _query_corpus(query_text: str) -> IndexedCorpus:
    tokens = query_text.split()
    inv: dict[str, list[tuple[int, int]]] = {
        term: [(0, tf)] for term, tf in Counter(tokens).items()
    }
    return IndexedCorpus(
        documents=[Document("__query__", query_text, "", metadata={"is_query": True})],
        processed_texts=[query_text],
        inverted_index=inv,
        vocabulary=sorted(inv.keys()),
    )


@pytest.fixture
def fitted_retriever(in_memory_store, in_memory_repo):
    """A retriever fitted on a 3-doc base corpus with shared vocabulary."""
    corpus = _build_indexed_corpus(
        [
            ("d1", "u1", ["diabetes", "glucosa", "insulina"]),
            ("d2", "u2", ["hipertension", "presion", "arterial"]),
            ("d3", "u3", ["diabetes", "insulina", "pancreas"]),
        ]
    )
    retriever = LSIRetriever(
        repository=in_memory_repo,
        document_store=in_memory_store,
        n_components=2,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)
    return retriever


class TestAddDocuments:
    def test_returns_number_added(self, fitted_retriever):
        new_corpus = _build_indexed_corpus(
            [("d4", "u4", ["diabetes", "glucosa", "pancreas"])]
        )
        assert fitted_retriever.add_documents(new_corpus) == 1

    def test_empty_corpus_returns_zero(self, fitted_retriever):
        empty = IndexedCorpus(
            documents=[], processed_texts=[], inverted_index={}, vocabulary=[]
        )
        assert fitted_retriever.add_documents(empty) == 0

    def test_before_fit_raises(self, in_memory_store, in_memory_repo):
        retriever = LSIRetriever(
            repository=in_memory_repo, document_store=in_memory_store
        )
        new_corpus = _build_indexed_corpus([("d4", "u4", ["diabetes"])])
        with pytest.raises(RuntimeError):
            retriever.add_documents(new_corpus)

    def test_added_document_is_persisted(self, fitted_retriever, in_memory_store):
        new_corpus = _build_indexed_corpus(
            [("d4", "u4", ["diabetes", "glucosa"])]
        )
        fitted_retriever.add_documents(new_corpus)
        assert in_memory_store.exists("d4")

    def test_does_not_refit_the_svd_model(self, fitted_retriever):
        """Folding-in must reuse the trained SVD, never re-fit it."""
        svd_before = fitted_retriever.model._svd
        new_corpus = _build_indexed_corpus(
            [("d4", "u4", ["diabetes", "glucosa", "pancreas"])]
        )
        fitted_retriever.add_documents(new_corpus)
        # Same SVD object identity → no rebuild happened.
        assert fitted_retriever.model._svd is svd_before


class TestRetrievalAfterFoldingIn:
    def test_folded_in_document_is_retrievable(self, fitted_retriever):
        new_corpus = _build_indexed_corpus(
            [("d4", "u4", ["diabetes", "glucosa", "pancreas"])]
        )
        fitted_retriever.add_documents(new_corpus)

        query = Query(
            text="diabetes glucosa pancreas",
            indexed_corpus=_query_corpus("diabetes glucosa pancreas"),
        )
        results = fitted_retriever.retrieve(query, top_k=5)
        assert any(r.document.doc_id == "d4" for r in results), (
            "folded-in document d4 was not retrieved"
        )

    def test_oov_only_document_does_not_crash(self, fitted_retriever, in_memory_store):
        """A doc whose terms are all out-of-vocabulary folds in as a zero vector."""
        oov_corpus = _build_indexed_corpus(
            [("d_oov", "u", ["palabranueva", "otroterminonuevo"])]
        )
        added = fitted_retriever.add_documents(oov_corpus)
        assert added == 1
        assert in_memory_store.exists("d_oov")


class TestRebalanceTracking:
    def test_incremental_fraction_after_add(self, fitted_retriever):
        # Base = 3 docs; add 1 → fraction 1/4 = 0.25
        new_corpus = _build_indexed_corpus(
            [("d4", "u4", ["diabetes", "glucosa"])]
        )
        fitted_retriever.add_documents(new_corpus)
        assert fitted_retriever.incremental_fraction == pytest.approx(0.25)

    def test_needs_rebalance_triggers_past_threshold(self, fitted_retriever):
        # 1 added over base 3 → 0.25 > 0.2 default threshold
        fitted_retriever.add_documents(
            _build_indexed_corpus([("d4", "u4", ["diabetes"])])
        )
        assert fitted_retriever.needs_rebalance() is True
        # But not past a higher threshold
        assert fitted_retriever.needs_rebalance(threshold=0.5) is False

    def test_fit_resets_counters(self, fitted_retriever):
        fitted_retriever.add_documents(
            _build_indexed_corpus([("d4", "u4", ["diabetes"])])
        )
        assert fitted_retriever.incremental_fraction > 0
        # A full refit ("balanceo") resets the dynamic-indexing counters.
        rebuilt = _build_indexed_corpus(
            [
                ("d1", "u1", ["diabetes", "glucosa", "insulina"]),
                ("d2", "u2", ["hipertension", "presion", "arterial"]),
            ]
        )
        fitted_retriever.fit(rebuilt)
        assert fitted_retriever.incremental_fraction == 0.0
        assert fitted_retriever.needs_rebalance() is False
