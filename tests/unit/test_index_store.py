"""Tests for IndexStore — persistence of IndexedCorpus and spell vocabulary."""

from __future__ import annotations

import json

import pytest

from core.interfaces import IndexedCorpus
from core.models import Document
from modules.indexer.index_store import IndexStore, IndexStoreError
from modules.text_processor.spell_checker import TrieSpellChecker


@pytest.fixture
def store(tmp_path) -> IndexStore:
    return IndexStore(storage_dir=tmp_path / "indexer")


@pytest.fixture
def small_corpus() -> IndexedCorpus:
    return IndexedCorpus(
        documents=[
            Document("d1", "alpha beta", "u1"),
            Document("d2", "beta gamma", "u2"),
        ],
        processed_texts=["alpha beta", "beta gamma"],
        inverted_index={"alpha": [(0, 1)], "beta": [(0, 1), (1, 1)], "gamma": [(1, 1)]},
        vocabulary=["alpha", "beta", "gamma"],
    )


class TestSaveLoadRoundTrip:
    def test_round_trip_preserves_corpus(self, store, small_corpus):
        store.save(small_corpus)
        loaded = store.load()
        assert [d.doc_id for d in loaded.documents] == ["d1", "d2"]
        assert loaded.vocabulary == small_corpus.vocabulary
        assert loaded.inverted_index == small_corpus.inverted_index

    def test_load_without_save_raises(self, store):
        with pytest.raises(IndexStoreError):
            store.load()

    def test_exists_reflects_save(self, store, small_corpus):
        assert store.exists() is False
        store.save(small_corpus)
        assert store.exists() is True


class TestManifest:
    def test_manifest_after_save(self, store, small_corpus):
        store.save(small_corpus)
        manifest = store.manifest()
        assert manifest["n_documents"] == 2
        assert manifest["n_terms"] == 3
        assert "created_at" in manifest
        assert "updated_at" in manifest
        assert manifest["schema_version"] == "1.0"

    def test_manifest_preserves_created_at(self, store, small_corpus):
        store.save(small_corpus)
        first_manifest = store.manifest()
        # Save again — created_at must remain
        store.save(small_corpus)
        second_manifest = store.manifest()
        assert second_manifest["created_at"] == first_manifest["created_at"]

    def test_manifest_empty_when_no_save(self, store):
        assert store.manifest() == {}


class TestIndexedDocIds:
    def test_indexed_doc_ids_after_save(self, store, small_corpus):
        store.save(small_corpus)
        assert store.indexed_doc_ids() == {"d1", "d2"}

    def test_indexed_doc_ids_empty_when_unsaved(self, store):
        assert store.indexed_doc_ids() == set()


class TestSpellVocabulary:
    def test_save_and_restore_vocabulary(self, store):
        checker = TrieSpellChecker()
        for w in ("hipertensión", "diabetes", "asma"):
            checker._insert(w)
        store.save_spell_vocabulary(checker)

        restored = TrieSpellChecker()
        n = store.load_spell_vocabulary(restored)
        assert n == 3
        assert set(restored.words()) == {"hipertensión", "diabetes", "asma"}

    def test_load_spell_vocabulary_when_missing_returns_zero(self, store):
        n = store.load_spell_vocabulary(TrieSpellChecker())
        assert n == 0


class TestClear:
    def test_clear_removes_all_artifacts(self, store, small_corpus):
        checker = TrieSpellChecker()
        checker._insert("alpha")
        store.save(small_corpus)
        store.save_spell_vocabulary(checker)
        assert store.exists()

        store.clear()
        assert store.exists() is False
        assert store.manifest() == {}
        assert store.indexed_doc_ids() == set()
