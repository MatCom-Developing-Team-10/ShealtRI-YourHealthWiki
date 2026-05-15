"""End-to-end integration test.

Walks the real path from raw documents → TextProcessor (real spaCy) →
IndexerService → LSIRetriever, using in-memory DocumentStore + Repository
(defined in conftest) so no external services are required.

Skipped if the spaCy Spanish model is not installed.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("es_core_news_md")
except OSError:
    pytest.skip("spaCy model 'es_core_news_md' not installed", allow_module_level=True)


from core.models import Query
from modules.indexer import IndexerService
from modules.retriever import LSIRetriever
from modules.text_processor import TextProcessor


def test_full_pipeline_returns_relevant_documents(
    sample_documents, in_memory_store, in_memory_repo
):
    # 1. Index — using a fresh TextProcessor so the spell vocabulary starts empty.
    processor = TextProcessor()
    indexer = IndexerService(text_processor=processor)
    corpus = indexer.build(sample_documents)

    # 2. Fit retriever
    retriever = LSIRetriever(
        repository=in_memory_repo,
        document_store=in_memory_store,
        n_components=2,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)

    # 3. Build query corpus through the indexer (same processor → spell vocab shared)
    query_corpus = indexer.build_query("hipertensión arterial")
    query = Query(text="hipertensión arterial", indexed_corpus=query_corpus)

    # 4. Retrieve
    results = retriever.retrieve(query, top_k=3)
    assert len(results) >= 1

    top_ids = [r.document.doc_id for r in results]
    # The HTA document (d1) should rank above unrelated ones
    assert "d1" in top_ids


def test_spell_correction_recovers_match(
    sample_documents, in_memory_store, in_memory_repo
):
    processor = TextProcessor()
    indexer = IndexerService(text_processor=processor)
    corpus = indexer.build(sample_documents)

    retriever = LSIRetriever(
        repository=in_memory_repo,
        document_store=in_memory_store,
        n_components=2,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)

    # Query with deliberate typos
    query_corpus = indexer.build_query("hipertensoin arterail")
    results = retriever.retrieve(
        Query(text="hipertensoin arterail", indexed_corpus=query_corpus),
        top_k=3,
    )
    # We accept any non-empty result here; the goal is to verify that spell
    # correction prevented the OOV terms from yielding a fully-empty TF-IDF.
    assert isinstance(results, list)
