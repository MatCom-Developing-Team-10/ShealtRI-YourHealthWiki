"""Integration test: TextProcessor (real spaCy) <-> IndexerService.

Verifies that the indexer correctly uses the TextProcessor instance to:
    1. populate the spell-checker vocabulary during ``build()``
    2. consult that same vocabulary during ``build_query()`` so misspelled
       query tokens get corrected to known terms BEFORE the inverted index
       is constructed for the query.

This is the contract the README documents but no unit test alone can
verify because both sides of the boundary (lemmatisation + trie) must
agree on the post-processing form of each token.

Uses the ``fresh_processor`` fixture (defined in conftest) to share the
session's spaCy model while still starting each test with an empty
spell-checker vocabulary.
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


class TestSpellVocabPopulation:
    def test_build_populates_spell_checker(self, fresh_processor):
        indexer = IndexerService(text_processor=fresh_processor)

        # The spell checker starts empty
        assert fresh_processor.spell_checker.words() == []

        indexer.build(
            [
                _doc("d1", "hipertensión arterial crónica"),
                _doc("d2", "diabetes mellitus tipo dos"),
            ]
        )

        vocab = set(fresh_processor.spell_checker.words())
        assert vocab, "spell-checker vocabulary stayed empty after build()"
        # Spot check: at least one medical-term stem must be present
        assert any(any(stem in w for stem in ("hipertensi", "diabet")) for w in vocab)

    def test_build_query_uses_existing_vocab_for_correction(self, fresh_processor):
        indexer = IndexerService(text_processor=fresh_processor)
        indexer.build(
            [
                _doc("d1", "hipertensión arterial"),
                _doc("d2", "diabetes mellitus glucosa insulina"),
            ]
        )

        # 'diabets' is distance 1 from 'diabetes' (which is in the vocab)
        qc = indexer.build_query("diabets glucosa")
        # The literal typo must not survive into the query vocabulary
        assert "diabets" not in qc.vocabulary


class TestSharedTextProcessor:
    def test_two_indexers_sharing_processor_share_spell_vocab(self, fresh_processor):
        """Two indexers wired to the same TextProcessor share spell vocab."""
        indexer_a = IndexerService(text_processor=fresh_processor)
        indexer_b = IndexerService(text_processor=fresh_processor)

        indexer_a.build([_doc("d1", "hipertensión arterial")])
        # indexer_b's query uses the shared spell checker, populated by indexer_a
        qc = indexer_b.build_query("hipertensoin arterail")
        assert "hipertensoin" not in qc.vocabulary
