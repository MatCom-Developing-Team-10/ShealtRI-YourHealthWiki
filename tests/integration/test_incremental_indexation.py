"""Integration test: incremental update + remove flow.

The end-user crawls daily. We need to verify that:
    - update() adds new documents WITHOUT re-processing the whole corpus
    - update() is idempotent on duplicates (same crawl ran twice is a no-op)
    - remove() drops a document and all its terms become unreachable
    - stats() reflects the new state after each step

These properties are checked at unit level (with a fake processor) but here
we exercise them with the real TextProcessor + spaCy lemmatiser so that any
discrepancy between fake-tokens and lemmatised tokens is caught.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("es_core_news_md")
except OSError:
    pytest.skip("spaCy model 'es_core_news_md' not installed", allow_module_level=True)


from core.models import Document
from modules.indexer.service import IndexerService


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, text=text, url=f"http://test/{doc_id}")


@pytest.fixture
def initial_corpus(fresh_processor):
    """Initial 2-doc corpus indexed with a fresh spell-checker."""
    indexer = IndexerService(text_processor=fresh_processor)
    docs = [
        _doc("d_hta", "hipertensión arterial enfermedad cardiovascular"),
        _doc("d_dm", "diabetes mellitus glucosa insulina páncreas"),
    ]
    corpus = indexer.build(docs)
    return fresh_processor, indexer, corpus


class TestUpdate:
    def test_update_extends_corpus(self, initial_corpus):
        _, indexer, corpus = initial_corpus
        before = len(corpus.documents)
        new_doc = _doc("d_asma", "asma bronquial enfermedad respiratoria")
        updated = indexer.update(corpus, [new_doc])

        assert len(updated.documents) == before + 1
        assert any(d.doc_id == "d_asma" for d in updated.documents)

    def test_update_does_not_mutate_input(self, initial_corpus):
        _, indexer, corpus = initial_corpus
        original_count = len(corpus.documents)
        indexer.update(corpus, [_doc("d_new", "nuevo documento médico")])
        assert len(corpus.documents) == original_count

    def test_update_is_idempotent_on_duplicate_ids(self, initial_corpus):
        _, indexer, corpus = initial_corpus
        # Re-submit a doc with the same id but different text
        dup = _doc("d_hta", "completely different text about something else")
        updated = indexer.update(corpus, [dup])
        assert len(updated.documents) == len(corpus.documents)

    def test_update_grows_inverted_index_for_truly_new_terms(self, initial_corpus):
        _, indexer, corpus = initial_corpus
        new_doc = _doc(
            "d_alz",
            "alzheimer demencia neurodegenerativa memoria cognición",
        )
        updated = indexer.update(corpus, [new_doc])
        new_terms = set(updated.vocabulary) - set(corpus.vocabulary)
        assert new_terms, "update introduced no new vocabulary terms"


class TestRemove:
    def test_remove_drops_document_from_corpus(self, initial_corpus):
        _, indexer, corpus = initial_corpus
        reduced = indexer.remove(corpus, ["d_hta"])
        assert all(d.doc_id != "d_hta" for d in reduced.documents)

    def test_remove_drops_terms_unique_to_removed_doc(self, initial_corpus):
        _, indexer, corpus = initial_corpus
        # Add a doc using real short medical Spanish terms that the lemmatiser
        # keeps under max_token_length (20). 'osteoporosis' and 'densitometría'
        # do not appear in the initial corpus (HTA + diabetes) so they should
        # show up as new vocabulary entries after update().
        unique_doc = _doc(
            "d_unique",
            "osteoporosis densitometría bifosfonato calcio",
        )
        plus_one = indexer.update(corpus, [unique_doc])
        had_unique_terms = set(plus_one.vocabulary) - set(corpus.vocabulary)
        assert had_unique_terms, (
            "Update added a doc with novel terms but no new vocabulary appeared. "
            f"plus_one.vocabulary = {plus_one.vocabulary}"
        )

        reduced = indexer.remove(plus_one, ["d_unique"])
        gone = had_unique_terms - set(reduced.vocabulary)
        # Terms that existed ONLY in the removed doc must disappear
        assert gone == had_unique_terms

    def test_remove_renumbers_indices(self, initial_corpus):
        _, indexer, corpus = initial_corpus
        # Add a third doc, then remove the middle one
        extended = indexer.update(
            corpus,
            [_doc("d_asma", "asma bronquial respiratoria crónica")],
        )
        reduced = indexer.remove(extended, [extended.documents[1].doc_id])

        # All postings indices must be valid for the new (smaller) doc list
        for term, postings in reduced.inverted_index.items():
            for doc_idx, _ in postings:
                assert 0 <= doc_idx < len(reduced.documents), (
                    f"posting for {term!r} points to invalid index {doc_idx}"
                )


class TestStats:
    def test_stats_changes_with_lifecycle(self, initial_corpus):
        _, indexer, corpus = initial_corpus
        s0 = IndexerService.stats(corpus)

        updated = indexer.update(
            corpus, [_doc("d_extra", "ejemplo adicional sobre un tema cualquiera")]
        )
        s1 = IndexerService.stats(updated)
        assert s1["n_documents"] == s0["n_documents"] + 1

        reduced = indexer.remove(updated, ["d_extra"])
        s2 = IndexerService.stats(reduced)
        assert s2["n_documents"] == s0["n_documents"]
