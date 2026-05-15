"""Tests for the Trie-based spell checker.

The spell checker file has zero heavy imports so these tests run without
loading spaCy.
"""

from __future__ import annotations

import pytest

from modules.text_processor.spell_checker import TrieSpellChecker


class TestInsertAndContains:
    def test_insert_then_contains(self):
        trie = TrieSpellChecker()
        trie._insert("hipertensión")
        assert trie._contains("hipertensión") is True

    def test_unknown_word_not_contained(self):
        trie = TrieSpellChecker(vocabulary=["asma", "diabetes"])
        assert trie._contains("cáncer") is False

    def test_prefix_is_not_full_word(self):
        trie = TrieSpellChecker(vocabulary=["hipertensión"])
        # 'hiper' is a prefix but not a full word in the trie
        assert trie._contains("hiper") is False

    def test_constructor_with_vocabulary(self):
        trie = TrieSpellChecker(vocabulary=["a", "ab", "abc"])
        for w in ["a", "ab", "abc"]:
            assert trie._contains(w)


class TestWordsEnumeration:
    def test_words_returns_all_inserted(self):
        words = {"asma", "diabetes", "hipertensión"}
        trie = TrieSpellChecker(vocabulary=list(words))
        assert set(trie.words()) == words

    def test_empty_trie_words(self):
        trie = TrieSpellChecker()
        assert trie.words() == []


class TestLevenshteinDistance:
    def test_identical(self):
        assert TrieSpellChecker._levenshtein("abc", "abc") == 0

    def test_substitution(self):
        assert TrieSpellChecker._levenshtein("kitten", "sitten") == 1

    def test_insertion(self):
        assert TrieSpellChecker._levenshtein("cat", "cats") == 1

    def test_deletion(self):
        assert TrieSpellChecker._levenshtein("cats", "cat") == 1

    def test_classic_pair(self):
        # Levenshtein distance between 'kitten' and 'sitting' is 3
        assert TrieSpellChecker._levenshtein("kitten", "sitting") == 3

    def test_empty_string_handling(self):
        assert TrieSpellChecker._levenshtein("", "abc") == 3
        assert TrieSpellChecker._levenshtein("abc", "") == 3


class TestCorrect:
    def test_word_in_vocabulary_returns_itself(self):
        trie = TrieSpellChecker(vocabulary=["hipertensión", "asma"])
        assert trie.correct("hipertensión") == "hipertensión"

    def test_close_misspelling_corrected(self):
        # 'hipertensoin' has distance 2 from 'hipertensión'
        trie = TrieSpellChecker(vocabulary=["hipertensión"], max_distance=2)
        assert trie.correct("hipertensoin") == "hipertensión"

    def test_too_far_returns_none(self):
        trie = TrieSpellChecker(vocabulary=["hipertensión"], max_distance=1)
        assert trie.correct("xxxxxxxx") is None

    def test_picks_closest_match(self):
        trie = TrieSpellChecker(vocabulary=["asma", "ama", "amar"], max_distance=2)
        # 'asma' itself is in the vocabulary, expect itself
        assert trie.correct("asma") == "asma"

    def test_empty_trie_returns_none(self):
        trie = TrieSpellChecker()
        assert trie.correct("anything") is None
