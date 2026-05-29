"""End-to-end integration test using the in-memory fakes.

Walks the real path from raw documents → TextProcessor (real spaCy) →
IndexerService → LSIRetriever, using ``InMemoryDocumentStore`` +
``InMemoryRepository`` from conftest (so no ChromaDB or filesystem
dependency is required).

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
from modules.indexer.service import IndexerConfig
from modules.retriever import LSIRetriever


# ---------------------------------------------------------------------------
# Heuristic: pick a doc_id from the 20-doc corpus that contains a substring
# in its raw text. The synthetic corpus assigns ids like 'doc_hipertension_001'
# so we don't hard-code them and instead query by topic.
# ---------------------------------------------------------------------------


def _topic_ids(documents, keyword: str) -> set[str]:
    """Return doc_ids whose raw text contains ``keyword`` (case-insensitive)."""
    kw = keyword.lower()
    return {d.doc_id for d in documents if kw in d.text.lower()}


def test_full_pipeline_returns_relevant_documents(
    sample_documents, in_memory_store, in_memory_repo, fresh_processor
):
    """An end-to-end query for 'hipertensión arterial' must surface HTA docs.

    The 20-doc corpus has at least one document specifically about
    hipertensión arterial; the LSI projection must rank it among the top-5.
    """
    indexer = IndexerService(
        text_processor=fresh_processor,
        config=IndexerConfig(min_term_frequency=1),
    )
    corpus = indexer.build(sample_documents)

    retriever = LSIRetriever(
        repository=in_memory_repo,
        document_store=in_memory_store,
        n_components=10,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)

    query_corpus = indexer.build_query("hipertensión arterial presión")
    query = Query(text="hipertensión arterial presión", indexed_corpus=query_corpus)

    results = retriever.retrieve(query, top_k=5)
    assert results, "retrieval returned no results for a topical query"

    top_ids = [r.document.doc_id for r in results]
    expected = _topic_ids(sample_documents, "hipertensión")
    assert expected, "fixture missing — no document about hipertensión"
    # At least one document on the topic must appear in the top-5
    assert set(top_ids) & expected, (
        f"top-5 {top_ids} contains no document about hipertensión "
        f"(expected any of {expected})"
    )


def test_spell_correction_recovers_match(
    sample_documents, in_memory_store, in_memory_repo, fresh_processor
):
    """A query with typos must still produce results thanks to spell correction.

    The processor's spell checker accumulates the corpus vocabulary during
    ``build()``; on ``build_query()`` it corrects close-distance typos back
    to known vocabulary so the TF-IDF transform doesn't drop everything.
    """
    indexer = IndexerService(
        text_processor=fresh_processor,
        config=IndexerConfig(min_term_frequency=1),
    )
    corpus = indexer.build(sample_documents)

    retriever = LSIRetriever(
        repository=in_memory_repo,
        document_store=in_memory_store,
        n_components=10,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)

    # 'hipertensoin arterail' has edit distance 2 from 'hipertensión arterial'
    query_corpus = indexer.build_query("hipertensoin arterail")
    results = retriever.retrieve(
        Query(text="hipertensoin arterail", indexed_corpus=query_corpus),
        top_k=5,
    )
    # The spell-correction layer must keep the query non-empty; the actual
    # ranking is asserted in tests/regression/test_ranking_golden.py.
    assert isinstance(results, list)
    assert results, "spell-corrected query collapsed to empty TF-IDF"
