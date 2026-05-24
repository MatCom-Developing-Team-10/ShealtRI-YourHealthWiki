"""Tests for the IndexerService.

These tests inject a fake TextProcessor that does deterministic whitespace
tokenization. This isolates the indexing logic from spaCy so the tests are
fast, deterministic, and independent of the lemmatizer.
"""

from __future__ import annotations

import pytest

from core.interfaces import IndexedCorpus
from core.models import Document
from modules.indexer.service import IndexerConfig, IndexerService
from modules.text_processor.spell_checker import TrieSpellChecker


# ---------------------------------------------------------------------------
# Fake TextProcessor
# ---------------------------------------------------------------------------


class _FakeTextProcessor:
    """Minimal stand-in for TextProcessor.

    Implements only the interface IndexerService relies on:
        - .process(text, is_query) -> str
        - .spell_checker (any object with words()/correct())
    """

    def __init__(self) -> None:
        self.spell_checker = TrieSpellChecker()
        self.calls: list[tuple[str, bool]] = []

    def process(self, text: str, is_query: bool = False) -> str:
        self.calls.append((text, is_query))
        text = text.strip().lower()
        if not text:
            return ""
        tokens = text.split()
        if is_query:
            corrected = []
            for tok in tokens:
                fix = self.spell_checker.correct(tok)
                corrected.append(fix if fix else tok)
            tokens = corrected
        else:
            for tok in tokens:
                self.spell_checker._insert(tok)
        return " ".join(tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, text=text, url=f"http://x/{doc_id}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuild:
    def test_returns_indexed_corpus(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        out = idx.build([_doc("d1", "hipertensión arterial")])
        assert isinstance(out, IndexedCorpus)
        assert len(out) == 1

    def test_empty_document_list(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        out = idx.build([])
        assert len(out) == 0
        assert out.vocabulary == []
        assert out.inverted_index == {}

    def test_inverted_index_structure(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        docs = [
            _doc("d1", "hta diabetes"),
            _doc("d2", "diabetes asma"),
        ]
        corpus = idx.build(docs)

        # 'diabetes' must appear in postings for both docs
        diabetes_postings = corpus.inverted_index["diabetes"]
        doc_indices = {idx_pair[0] for idx_pair in diabetes_postings}
        assert doc_indices == {0, 1}

    def test_term_frequency_counted(self):
        cfg = IndexerConfig(min_term_frequency=1)
        idx = IndexerService(text_processor=_FakeTextProcessor(), config=cfg)
        out = idx.build([_doc("d1", "asma asma asma diabetes")])
        # 'asma' appears 3 times in doc 0
        assert out.inverted_index["asma"] == [(0, 3)]
        assert out.inverted_index["diabetes"] == [(0, 1)]

    def test_vocabulary_is_sorted(self):
        cfg = IndexerConfig(min_term_frequency=1)
        idx = IndexerService(text_processor=_FakeTextProcessor(), config=cfg)
        out = idx.build(
            [_doc("d1", "zeta beta alpha"), _doc("d2", "delta gamma")]
        )
        assert out.vocabulary == sorted(out.vocabulary)

    def test_min_document_length_drops_short(self):
        cfg = IndexerConfig(min_document_length=3)
        idx = IndexerService(text_processor=_FakeTextProcessor(), config=cfg)
        out = idx.build(
            [
                _doc("short", "uno dos"),  # only 2 tokens — dropped
                _doc("long", "uno dos tres cuatro"),  # 4 tokens — kept
            ]
        )
        assert [d.doc_id for d in out.documents] == ["long"]

    def test_min_term_frequency_drops_rare(self):
        cfg = IndexerConfig(min_term_frequency=2)
        idx = IndexerService(text_processor=_FakeTextProcessor(), config=cfg)
        out = idx.build(
            [
                _doc("d1", "common common rare"),
                _doc("d2", "common other"),
            ]
        )
        assert "common" in out.vocabulary
        # 'rare' appears only once total → dropped
        assert "rare" not in out.vocabulary

    def test_processed_texts_match_documents_count(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        out = idx.build([_doc("d1", "a b"), _doc("d2", "c d")])
        assert len(out.processed_texts) == len(out.documents)


class TestUpdate:
    def test_appends_new_documents(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        existing = idx.build([_doc("d1", "alpha beta")])
        updated = idx.update(existing, [_doc("d2", "beta gamma")])
        assert len(updated) == 2
        assert "gamma" in updated.vocabulary

    def test_skips_duplicate_doc_ids(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        existing = idx.build([_doc("d1", "alpha")])
        updated = idx.update(existing, [_doc("d1", "different text")])
        assert len(updated) == 1

    def test_no_new_documents_returns_existing(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        existing = idx.build([_doc("d1", "a b")])
        same = idx.update(existing, [])
        assert same is existing

    def test_does_not_mutate_existing(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        existing = idx.build([_doc("d1", "a b")])
        original_doc_count = len(existing.documents)
        original_vocab = list(existing.vocabulary)
        idx.update(existing, [_doc("d2", "c d")])
        assert len(existing.documents) == original_doc_count
        assert list(existing.vocabulary) == original_vocab


class TestRemove:
    def test_removes_documents_and_renumbers_indices(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        corpus = idx.build(
            [
                _doc("d1", "alpha beta"),
                _doc("d2", "beta gamma"),
                _doc("d3", "delta epsilon"),
            ]
        )
        out = idx.remove(corpus, ["d2"])
        assert [d.doc_id for d in out.documents] == ["d1", "d3"]
        # 'gamma' was only in d2 → must be dropped from vocabulary
        assert "gamma" not in out.vocabulary

    def test_unknown_ids_silently_ignored(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        corpus = idx.build([_doc("d1", "alpha")])
        out = idx.remove(corpus, ["does-not-exist"])
        assert len(out.documents) == 1

    def test_empty_id_list_is_noop(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        corpus = idx.build([_doc("d1", "alpha")])
        same = idx.remove(corpus, [])
        assert same is corpus


class TestStats:
    def test_basic_counts(self):
        cfg = IndexerConfig(min_term_frequency=1)
        idx = IndexerService(text_processor=_FakeTextProcessor(), config=cfg)
        corpus = idx.build([_doc("d1", "a b c"), _doc("d2", "b c d")])
        stats = IndexerService.stats(corpus)
        assert stats["n_documents"] == 2
        assert stats["n_terms"] == 4  # a, b, c, d
        assert stats["total_tokens"] == 6
        assert stats["avg_tokens_per_doc"] == pytest.approx(3.0)

    def test_empty_corpus_safe(self):
        empty = IndexedCorpus(
            documents=[], processed_texts=[], inverted_index={}, vocabulary=[]
        )
        stats = IndexerService.stats(empty)
        assert stats["n_documents"] == 0
        assert stats["avg_tokens_per_doc"] == 0.0


class TestBuildQuery:
    def test_query_corpus_has_one_document(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        out = idx.build_query("hipertensión arterial")
        assert len(out) == 1
        assert out.documents[0].metadata.get("is_query") is True

    def test_empty_query_returns_empty_corpus(self):
        idx = IndexerService(text_processor=_FakeTextProcessor())
        out = idx.build_query("")
        assert out.vocabulary == []
        assert out.inverted_index == {}
        assert len(out) == 1  # placeholder document still present

    def test_query_terms_corrected_from_vocabulary(self):
        tp = _FakeTextProcessor()
        idx = IndexerService(text_processor=tp)
        # Build vocabulary from documents
        idx.build([_doc("d1", "hipertensión arterial diabetes")])
        # Query with typo
        out = idx.build_query("hipertensoin diabetes")
        # 'hipertensoin' should be corrected to a known vocabulary term
        assert "hipertensoin" not in out.vocabulary
