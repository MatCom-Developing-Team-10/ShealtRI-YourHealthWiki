"""Tests for the TextProcessor.

Uses the session-scoped ``text_processor`` fixture (defined in conftest.py)
so the spaCy model is loaded only once per test run.
"""

from __future__ import annotations

import pytest

# These tests require spaCy + the Spanish model.
spacy = pytest.importorskip("spacy")
try:
    spacy.load("es_core_news_md")
except OSError:
    pytest.skip("spaCy model 'es_core_news_md' not installed", allow_module_level=True)


from modules.text_processor import TextProcessor, TextProcessorConfig
from modules.text_processor.spell_checker import TrieSpellChecker


class TestNormalize:
    def test_lowercase(self, text_processor: TextProcessor):
        assert text_processor.normalize("HOLA") == "hola"

    def test_collapses_multiple_spaces(self, text_processor: TextProcessor):
        assert text_processor.normalize("hola    mundo") == "hola mundo"

    def test_strips_punctuation(self, text_processor: TextProcessor):
        out = text_processor.normalize("¿hola, mundo?")
        # Spanish accents preserved, punctuation gone
        assert "," not in out
        assert "?" not in out
        assert "hola" in out
        assert "mundo" in out

    def test_preserves_spanish_accents(self, text_processor: TextProcessor):
        out = text_processor.normalize("hipertensión")
        assert "ó" in out

    def test_empty_input(self, text_processor: TextProcessor):
        assert text_processor.normalize("") == ""

    def test_remove_accents_when_enabled(self):
        cfg = TextProcessorConfig(remove_accents=True)
        tp = TextProcessor(cfg)
        out = tp.normalize("hipertensión")
        assert "ó" not in out
        assert "hipertension" in out


class TestTokenize:
    def test_returns_list_of_strings(self, text_processor: TextProcessor):
        tokens = text_processor.tokenize("hipertensión arterial")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        assert "hipertensión" in tokens
        assert "arterial" in tokens


class TestRemoveStopwords:
    def test_drops_known_stopwords(self, text_processor: TextProcessor):
        tokens = ["la", "hipertensión", "es", "una", "enfermedad"]
        out = text_processor.remove_stopwords(tokens)
        assert "la" not in out
        assert "es" not in out
        assert "una" not in out
        assert "hipertensión" in out
        assert "enfermedad" in out


class TestFilterTokens:
    def test_min_length_filter(self):
        cfg = TextProcessorConfig(min_token_length=3)
        tp = TextProcessor(cfg)
        out = tp.filter_tokens(["a", "ab", "abc", "abcd"])
        assert "a" not in out
        assert "ab" not in out
        assert "abc" in out
        assert "abcd" in out

    def test_max_length_filter(self):
        cfg = TextProcessorConfig(max_token_length=5)
        tp = TextProcessor(cfg)
        out = tp.filter_tokens(["short", "longerword"])
        assert "short" in out
        assert "longerword" not in out


class TestProcess:
    def test_full_pipeline_on_documents(self, text_processor: TextProcessor):
        # is_query=False → tokens get added to the spell-check vocabulary
        out = text_processor.process(
            "La hipertensión arterial causa cefalea", is_query=False
        )
        assert isinstance(out, str)
        # Stopwords gone
        assert " la " not in f" {out} "
        # At least the medical content survived
        assert "hipertensión" in out or "hipertensiónes" in out

    def test_empty_input_returns_empty(self, text_processor: TextProcessor):
        assert text_processor.process("") == ""
        assert text_processor.process("   ") == ""

    def test_documents_populate_spell_vocabulary(self):
        # Use a fresh processor so we can assert vocabulary build-up.
        tp = TextProcessor()
        tp.process("hipertensión arterial cefalea", is_query=False)
        words = set(tp.spell_checker.words())
        # Tokens that survive stopword + length filtering must be in vocabulary
        assert "hipertensión" in words or "hipertensiónes" in words
        assert any(w in words for w in ("arterial", "cefalea"))

    def test_query_path_corrects_misspelling(self):
        tp = TextProcessor()
        # First, build vocabulary from a document
        tp.process("hipertensión arterial", is_query=False)
        # Then query with a typo (distance 2 from 'hipertensión')
        corrected = tp.process("hipertensoin arterail", is_query=True)
        # We expect spell correction to map back to known vocabulary tokens
        # (the exact spelling depends on spaCy lemmatization).
        # At minimum, 'hipertensoin' must NOT survive verbatim if a closer
        # vocabulary word exists.
        assert "hipertensoin" not in corrected.split()


class TestStopwordsProperty:
    def test_returns_a_copy(self, text_processor: TextProcessor):
        sw = text_processor.stopwords
        sw.add("__test__")
        assert "__test__" not in text_processor.stopwords


class TestSpellCheckerOwnership:
    def test_each_processor_has_own_spell_checker(self):
        tp_a = TextProcessor()
        tp_b = TextProcessor()
        tp_a.process("hipertensión", is_query=False)
        # tp_b must NOT have absorbed tp_a's vocabulary
        assert "hipertensión" not in tp_b.spell_checker.words()

    def test_spell_checker_is_trie_instance(self, text_processor: TextProcessor):
        assert isinstance(text_processor.spell_checker, TrieSpellChecker)
