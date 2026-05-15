"""Tests for FileSystemDocumentStore."""

from __future__ import annotations

import json

import pytest

from core.models import Document
from modules.indexer.document_store import (
    DocumentReadError,
    DocumentStoreError,
    DocumentWriteError,
    FileSystemDocumentStore,
)


@pytest.fixture
def store(tmp_path) -> FileSystemDocumentStore:
    return FileSystemDocumentStore(storage_dir=str(tmp_path / "docs"))


class TestAddAndGet:
    def test_add_then_get_round_trip(self, store):
        doc = Document(
            doc_id="d1", text="content", url="http://x", metadata={"k": 1}
        )
        store.add_documents([doc])
        out = store.get_by_id("d1")
        assert out is not None
        assert out.doc_id == "d1"
        assert out.text == "content"
        assert out.url == "http://x"
        assert out.metadata == {"k": 1}

    def test_get_missing_returns_none(self, store):
        assert store.get_by_id("missing") is None

    def test_get_by_ids_preserves_order_skips_missing(self, store):
        docs = [
            Document("a", "ta", "ua"),
            Document("b", "tb", "ub"),
            Document("c", "tc", "uc"),
        ]
        store.add_documents(docs)
        out = store.get_by_ids(["c", "missing", "a"])
        assert [d.doc_id for d in out] == ["c", "a"]

    def test_overwrites_existing(self, store):
        store.add_documents([Document("d1", "old", "u")])
        store.add_documents([Document("d1", "new", "u")])
        assert store.get_by_id("d1").text == "new"


class TestExistsAndDelete:
    def test_exists(self, store):
        assert store.exists("d1") is False
        store.add_documents([Document("d1", "t", "u")])
        assert store.exists("d1") is True

    def test_delete_existing_returns_true(self, store):
        store.add_documents([Document("d1", "t", "u")])
        assert store.delete("d1") is True
        assert store.exists("d1") is False

    def test_delete_missing_returns_false(self, store):
        assert store.delete("missing") is False


class TestPathSanitization:
    def test_path_traversal_id_is_hashed(self, store, tmp_path):
        # Path-traversal attempt — must be hashed, never resolve outside the store dir
        evil = "../../etc/passwd"
        store.add_documents([Document(evil, "t", "u")])
        # The malicious file must NOT exist outside the storage dir
        assert not (tmp_path.parent.parent / "etc" / "passwd.json").exists()
        # But the document is still retrievable by its original ID
        out = store.get_by_id(evil)
        assert out is not None
        assert out.doc_id == evil

    def test_id_with_special_chars_hashed(self, store):
        weird = "doc:with*special?chars"
        store.add_documents([Document(weird, "t", "u")])
        assert store.get_by_id(weird) is not None

    def test_long_id_hashed(self, store):
        long_id = "x" * 500
        store.add_documents([Document(long_id, "t", "u")])
        assert store.get_by_id(long_id) is not None

    def test_safe_id_preserved_as_filename(self, store, tmp_path):
        store.add_documents([Document("simple-id_123", "t", "u")])
        files = list((tmp_path / "docs").glob("*.json"))
        assert any(f.stem == "simple-id_123" for f in files)


class TestSerializationErrors:
    def test_non_serializable_metadata_raises(self, store):
        bad = Document("d1", "t", "u", metadata={"bad": object()})
        with pytest.raises(DocumentWriteError):
            store.add_documents([bad])

    def test_corrupted_file_raises_read_error(self, store, tmp_path):
        # Write garbage to a doc file, then attempt to read
        doc_path = (tmp_path / "docs" / "broken.json")
        doc_path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(DocumentReadError):
            store.get_by_id("broken")
