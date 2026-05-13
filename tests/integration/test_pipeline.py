"""Integration tests for the full retrieval pipeline.

Tests the wiring: TextProcessor → IndexerService → LSIRetriever → RetrievalContext.
Uses real module instances (no mocks) and 20 synthetic Spanish medical documents.

Run with:
    python -m pytest tests/integration/test_pipeline.py -v -s
"""

from __future__ import annotations

import pytest

from core.models import Document, Query
from core.pipeline import RetrievalContext
from infra.chroma_repository import ChromaRepository
from modules.indexer.document_store import FileSystemDocumentStore
from modules.indexer.service import IndexerService
from modules.retriever.service import LSIRetriever
from modules.text_processor.service import TextProcessor


# ---------------------------------------------------------------------------
# Session-scoped fitted retriever (expensive to build — share across tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_pipeline(
    sample_documents: list[Document],
    text_processor: TextProcessor,
    tmp_path_factory,
):
    """Build and fit the full pipeline once for all tests in this module."""
    chroma_dir = str(tmp_path_factory.mktemp("chroma"))
    store_dir = str(tmp_path_factory.mktemp("store"))

    indexer = IndexerService(text_processor=text_processor)
    corpus = indexer.build(sample_documents)

    repository = ChromaRepository(
        persist_directory=chroma_dir,
        collection_name="test_collection",
    )
    document_store = FileSystemDocumentStore(storage_dir=store_dir)
    retriever = LSIRetriever(
        repository=repository,
        document_store=document_store,
        model_dir=str(tmp_path_factory.mktemp("models")),
        n_components=10,  # small for speed; 20 docs → max 19 meaningful components
        similarity_threshold=0.0,  # no filtering in tests — inspect raw scores
    )
    retriever.fit(corpus)

    return indexer, corpus, retriever


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndexerBuildsCorpus:
    def test_returns_indexed_corpus_with_all_documents(
        self, fitted_pipeline, sample_documents
    ):
        _, corpus, _ = fitted_pipeline
        assert len(corpus.documents) == len(sample_documents)

    def test_corpus_length_invariant(self, fitted_pipeline):
        """documents and processed_texts must stay aligned."""
        _, corpus, _ = fitted_pipeline
        assert len(corpus.documents) == len(corpus.processed_texts)

    def test_vocabulary_not_empty(self, fitted_pipeline):
        _, corpus, _ = fitted_pipeline
        assert len(corpus.vocabulary) > 0

    def test_corpus_statistics(self, fitted_pipeline):
        """Stats method returns expected fields and sane values; prints for CI logs."""
        _, corpus, _ = fitted_pipeline
        stats = IndexerService.stats(corpus)

        print(f"\nCorpus stats: {stats}")  # visible with pytest -s

        assert stats["n_documents"] == 20
        assert stats["n_terms"] > 50
        assert stats["avg_tokens_per_doc"] > 5
        assert stats["total_tokens"] > 0
        assert "avg_postings_per_term" in stats


class TestRetrieverFit:
    def test_fit_completes_without_error(self, fitted_pipeline):
        """LSIRetriever.fit() must not raise."""
        _, _, retriever = fitted_pipeline
        assert retriever.tfidf is not None
        assert retriever.model is not None

    def test_fit_raises_before_fitting(
        self,
        sample_documents: list[Document],
        text_processor: TextProcessor,
        tmp_chroma_dir: str,
        tmp_store_dir: str,
        tmp_path,
    ):
        """retrieve() on an unfitted retriever must raise RuntimeError."""
        indexer = IndexerService(text_processor=TextProcessor())
        corpus = indexer.build(sample_documents)
        query_corpus = indexer.build_query("diabetes")
        query = Query(text="diabetes", indexed_corpus=query_corpus)

        repo = ChromaRepository(persist_directory=tmp_chroma_dir)
        store = FileSystemDocumentStore(storage_dir=tmp_store_dir)
        unfitted = LSIRetriever(repository=repo, document_store=store)

        with pytest.raises(RuntimeError, match="fitted or loaded"):
            unfitted.retrieve(query)


class TestRetrieval:
    def test_retrieve_returns_results_for_medical_query(self, fitted_pipeline):
        indexer, _, retriever = fitted_pipeline
        query_corpus = indexer.build_query("hipertensión arterial presión")
        query = Query(text="hipertensión arterial presión", indexed_corpus=query_corpus)
        results = retriever.retrieve(query, top_k=5)
        assert len(results) > 0

    def test_retrieve_scores_in_valid_range(self, fitted_pipeline):
        """All similarity scores must be in [0.0, 1.0]."""
        indexer, _, retriever = fitted_pipeline
        query_corpus = indexer.build_query("diabetes glucosa insulina")
        query = Query(text="diabetes glucosa insulina", indexed_corpus=query_corpus)
        results = retriever.retrieve(query, top_k=10)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

    def test_retrieve_raises_on_missing_indexed_corpus(self, fitted_pipeline):
        """retrieve() must raise ValueError if indexed_corpus is None."""
        _, _, retriever = fitted_pipeline
        bad_query = Query(text="asma", indexed_corpus=None)
        with pytest.raises(ValueError, match="indexed_corpus"):
            retriever.retrieve(bad_query)

    def test_query_with_typo_processes_without_crash(self, fitted_pipeline):
        """A query with a minor Spanish typo should not crash the pipeline."""
        indexer, _, retriever = fitted_pipeline
        # "diabets" is a plausible typo for "diabetes"
        query_corpus = indexer.build_query("diabets glucosa")
        query = Query(text="diabets glucosa", indexed_corpus=query_corpus)
        results = retriever.retrieve(query, top_k=5)
        # Just checking it doesn't raise; results may or may not be populated
        assert isinstance(results, list)


class TestRetrievalContext:
    def test_strategy_wrapper_returns_same_results(self, fitted_pipeline):
        """RetrievalContext must delegate to the underlying retriever."""
        indexer, _, retriever = fitted_pipeline
        query_corpus = indexer.build_query("cáncer colon")
        query = Query(text="cáncer colon", indexed_corpus=query_corpus)

        direct = retriever.retrieve(query, top_k=5)
        context = RetrievalContext(strategy=retriever)
        via_context = context.execute_search(query, top_k=5)

        assert len(direct) == len(via_context)
        for a, b in zip(direct, via_context):
            assert a.document.doc_id == b.document.doc_id
            assert a.score == pytest.approx(b.score)
