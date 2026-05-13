"""Tests for LSIRetriever — orchestration of TF-IDF, LSI, and storage."""

from __future__ import annotations

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
        from collections import Counter
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
    from collections import Counter
    inv: dict[str, list[tuple[int, int]]] = {
        term: [(0, tf)] for term, tf in Counter(tokens).items()
    }
    return IndexedCorpus(
        documents=[Document("__query__", query_text, "", metadata={"is_query": True})],
        processed_texts=[query_text],
        inverted_index=inv,
        vocabulary=sorted(inv.keys()),
    )


class TestFit:
    def test_fit_populates_storage(self, in_memory_store, in_memory_repo):
        corpus = _build_indexed_corpus(
            [
                ("d1", "u1", ["alpha", "beta", "alpha"]),
                ("d2", "u2", ["beta", "gamma"]),
                ("d3", "u3", ["delta", "epsilon", "gamma"]),
            ]
        )
        retriever = LSIRetriever(
            repository=in_memory_repo,
            document_store=in_memory_store,
            n_components=2,
        )
        retriever.fit(corpus)

        # All docs persisted in both stores
        assert in_memory_store.exists("d1")
        assert in_memory_store.exists("d2")
        assert in_memory_store.exists("d3")
        # Sub-components fitted
        assert retriever.tfidf is not None
        assert retriever.model is not None


class TestRetrieveErrors:
    def test_retrieve_before_fit_raises(self, in_memory_store, in_memory_repo):
        retriever = LSIRetriever(
            repository=in_memory_repo, document_store=in_memory_store
        )
        with pytest.raises(RuntimeError):
            retriever.retrieve(Query(text="x", indexed_corpus=_query_corpus("x")))

    def test_query_without_indexed_corpus_raises(
        self, in_memory_store, in_memory_repo
    ):
        # Use 2 docs so the SVD has effective_k >= 1 (see LSIModel docs)
        corpus = _build_indexed_corpus(
            [
                ("d1", "u1", ["alpha", "beta"]),
                ("d2", "u2", ["beta", "gamma"]),
            ]
        )
        retriever = LSIRetriever(
            repository=in_memory_repo,
            document_store=in_memory_store,
            n_components=1,
        )
        retriever.fit(corpus)
        with pytest.raises(ValueError, match="indexed_corpus"):
            retriever.retrieve(Query(text="anything", indexed_corpus=None))


class TestRetrieve:
    def test_returns_top_k_results(self, in_memory_store, in_memory_repo):
        corpus = _build_indexed_corpus(
            [
                ("d1", "u1", ["alpha", "beta", "alpha"]),
                ("d2", "u2", ["beta", "gamma"]),
                ("d3", "u3", ["delta", "epsilon", "gamma"]),
            ]
        )
        retriever = LSIRetriever(
            repository=in_memory_repo,
            document_store=in_memory_store,
            n_components=2,
            similarity_threshold=0.0,  # no filtering
        )
        retriever.fit(corpus)
        query = Query(text="alpha beta", indexed_corpus=_query_corpus("alpha beta"))
        results = retriever.retrieve(query, top_k=2)
        assert len(results) <= 2
        # Results are sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_filters_low_scores(
        self, in_memory_store, in_memory_repo
    ):
        corpus = _build_indexed_corpus(
            [
                ("d1", "u1", ["alpha", "beta"]),
                ("d2", "u2", ["unrelated", "different"]),
            ]
        )
        retriever = LSIRetriever(
            repository=in_memory_repo,
            document_store=in_memory_store,
            n_components=1,
            similarity_threshold=0.99,  # extreme threshold filters most
        )
        retriever.fit(corpus)
        query = Query(text="zzz", indexed_corpus=_query_corpus("zzz"))
        results = retriever.retrieve(query)
        # 'zzz' is OOV so the query vector is zero → no match passes threshold
        assert results == []

    def test_results_contain_full_document_text(
        self, in_memory_store, in_memory_repo
    ):
        corpus = _build_indexed_corpus(
            [
                ("d1", "u1", ["alpha", "beta", "alpha"]),
                ("d2", "u2", ["gamma", "delta"]),
            ]
        )
        retriever = LSIRetriever(
            repository=in_memory_repo,
            document_store=in_memory_store,
            n_components=1,
            similarity_threshold=0.0,
        )
        retriever.fit(corpus)
        query = Query(text="alpha", indexed_corpus=_query_corpus("alpha"))
        results = retriever.retrieve(query)
        assert len(results) >= 1
        assert results[0].document.text  # non-empty text fetched from store

    def test_empty_search_results_returns_empty_list(
        self, in_memory_store, in_memory_repo, monkeypatch
    ):
        corpus = _build_indexed_corpus(
            [
                ("d1", "u1", ["alpha", "beta"]),
                ("d2", "u2", ["gamma", "delta"]),
            ]
        )
        retriever = LSIRetriever(
            repository=in_memory_repo,
            document_store=in_memory_store,
            n_components=1,
            similarity_threshold=0.0,
        )
        retriever.fit(corpus)

        # Force the repo to return nothing
        monkeypatch.setattr(in_memory_repo, "search_similar", lambda v, top_k: [])
        query = Query(text="alpha", indexed_corpus=_query_corpus("alpha"))
        assert retriever.retrieve(query) == []


class TestPersistence:
    def test_save_load_round_trip(
        self, in_memory_store, in_memory_repo, tmp_path
    ):
        corpus = _build_indexed_corpus(
            [
                ("d1", "u1", ["alpha", "beta"]),
                ("d2", "u2", ["beta", "gamma"]),
            ]
        )
        retriever = LSIRetriever(
            repository=in_memory_repo,
            document_store=in_memory_store,
            model_dir=str(tmp_path / "models"),
            n_components=1,
        )
        retriever.fit(corpus)
        retriever.save()

        loaded = LSIRetriever.load(
            repository=in_memory_repo,
            document_store=in_memory_store,
            model_dir=str(tmp_path / "models"),
        )
        assert loaded.tfidf is not None
        assert loaded.model is not None
