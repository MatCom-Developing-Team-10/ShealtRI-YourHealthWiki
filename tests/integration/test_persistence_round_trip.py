"""Integration test: full persistence round trip.

Covers the path:
    1. Index a corpus with IndexerService.
    2. Save IndexedCorpus + spell vocab via IndexStore.
    3. Fit and save LSI artifacts (TF-IDF + SVD) via LSIRetriever.save().
    4. Wipe in-memory state.
    5. Re-load everything from disk into fresh objects.
    6. Re-issue the same query and check that the top-k IDs are identical.

Without this test, a silent change in serialization format could ship
without anyone noticing until users tried to reload a saved index.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("es_core_news_md")
except OSError:
    pytest.skip("spaCy model 'es_core_news_md' not installed", allow_module_level=True)


from core.models import Query
from modules.indexer.index_store import IndexStore
from modules.indexer.service import IndexerService
from modules.retriever import LSIRetriever


def _top_ids(retriever, indexer, query_text: str, k: int = 5) -> list[str]:
    qc = indexer.build_query(query_text)
    results = retriever.retrieve(
        Query(text=query_text, indexed_corpus=qc), top_k=k
    )
    return [r.document.doc_id for r in results]


def test_index_round_trip_preserves_query_results(
    sample_documents, in_memory_store, in_memory_repo, tmp_path, fresh_processor
):
    """Index + save artifacts; reload; verify the same query yields the same top-k.

    The Repository here is an InMemoryRepository fed by the loaded LSI
    embeddings, so the only thing actually round-tripping on disk is:
        - IndexedCorpus (corpus.joblib)
        - spell-checker vocabulary (spell_vocab.txt)
        - TF-IDF processor (tfidf.joblib)
        - SVD model (svd.joblib)
    """
    index_dir = tmp_path / "indexer"
    model_dir = tmp_path / "models"

    # --- 1) Build & save ---
    indexer = IndexerService(text_processor=fresh_processor)
    corpus = indexer.build(sample_documents)

    retriever = LSIRetriever(
        repository=in_memory_repo,
        document_store=in_memory_store,
        model_dir=str(model_dir),
        n_components=10,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)

    store = IndexStore(storage_dir=index_dir)
    store.save(corpus)
    store.save_spell_vocabulary(fresh_processor.spell_checker)
    retriever.save()

    # Get the reference result BEFORE swapping anything.
    query = "hipertensión arterial"
    expected_ids = _top_ids(retriever, indexer, query, k=5)
    assert expected_ids, "baseline query returned no results"

    # --- 2) Wipe in-memory retriever state (keep processor — we'll reuse spaCy) ---
    del indexer, retriever

    # --- 3) Reload from disk into fresh objects ---
    # Reuse the same processor instance but clear and reload its spell vocab
    # from disk; this mimics a fresh start without re-loading spaCy.
    from modules.text_processor.spell_checker import TrieSpellChecker
    fresh_processor.spell_checker = TrieSpellChecker()
    n_words = store.load_spell_vocabulary(fresh_processor.spell_checker)
    assert n_words > 0, "spell vocabulary did not load"

    reloaded_corpus = store.load()
    assert len(reloaded_corpus.documents) == len(sample_documents)

    reloaded_indexer = IndexerService(text_processor=fresh_processor)

    reloaded_retriever = LSIRetriever.load(
        repository=in_memory_repo,
        document_store=in_memory_store,
        model_dir=str(model_dir),
        similarity_threshold=0.0,
    )

    # --- 4) Same query, compare ---
    got_ids = _top_ids(reloaded_retriever, reloaded_indexer, query, k=5)
    assert got_ids == expected_ids, (
        "Round-trip changed retrieval results.\n"
        f"  expected: {expected_ids}\n"
        f"  got:      {got_ids}"
    )


def test_manifest_metadata_after_reload(sample_documents, tmp_path, fresh_processor):
    """The manifest is the only source of truth for n_documents/n_terms after restart."""
    indexer = IndexerService(text_processor=fresh_processor)
    corpus = indexer.build(sample_documents)

    store = IndexStore(storage_dir=tmp_path / "indexer")
    store.save(corpus)

    manifest = store.manifest()
    assert manifest["n_documents"] == len(corpus.documents)
    assert manifest["n_terms"] == len(corpus.vocabulary)
    assert manifest["schema_version"] == "1.0"
    assert manifest["created_at"]
    assert manifest["updated_at"]

    # The doc_ids sidecar file lets you check membership without loading the corpus
    assert store.indexed_doc_ids() == {d.doc_id for d in corpus.documents}
