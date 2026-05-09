"""Tests for RawDocumentStorage (JSONL persistence for crawler output)."""

from __future__ import annotations

import json

import pytest

from core.models import Document
from infra.storage import RawDocumentStorage, RawStorageError


@pytest.fixture
def storage(tmp_path) -> RawDocumentStorage:
    return RawDocumentStorage(output_dir=str(tmp_path / "raw"))


class TestSourcePath:
    def test_normalizes_source_name(self, storage):
        path = storage.source_path("Mayo Clinic")
        assert path.name == "mayo_clinic.jsonl"

    def test_jsonl_extension(self, storage):
        path = storage.source_path("nhs")
        assert path.suffix == ".jsonl"


class TestSaveSingle:
    def test_appends_to_jsonl(self, storage):
        doc = Document("d1", "text", "https://x", metadata={"k": 1})
        storage.save(doc, "src")
        path = storage.source_path("src")
        with open(path, encoding="utf-8") as f:
            line = f.readline()
        record = json.loads(line)
        assert record["doc_id"] == "d1"
        assert record["text"] == "text"
        assert record["url"] == "https://x"
        assert record["metadata"] == {"k": 1}

    def test_non_serializable_metadata_raises(self, storage):
        doc = Document("d1", "t", "u", metadata={"bad": object()})
        with pytest.raises(RawStorageError):
            storage.save(doc, "src")


class TestSaveBatch:
    def test_returns_count_of_written(self, storage):
        docs = [
            Document("d1", "t1", "u1"),
            Document("d2", "t2", "u2"),
            Document("d3", "t3", "u3"),
        ]
        n = storage.save_batch(docs, "src")
        assert n == 3

    def test_skips_unserializable_continues_with_rest(self, storage):
        docs = [
            Document("d1", "t1", "u1"),
            Document("bad", "t", "u", metadata={"o": object()}),
            Document("d3", "t3", "u3"),
        ]
        n = storage.save_batch(docs, "src")
        assert n == 2

    def test_empty_batch_returns_zero(self, storage):
        assert storage.save_batch([], "src") == 0

    def test_appends_across_calls(self, storage):
        storage.save_batch([Document("d1", "t1", "u1")], "src")
        storage.save_batch([Document("d2", "t2", "u2")], "src")
        path = storage.source_path("src")
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["doc_id"] == "d1"
        assert json.loads(lines[1])["doc_id"] == "d2"

    def test_unicode_preserved(self, storage):
        storage.save_batch(
            [Document("d1", "hipertensión arterial", "u")], "src"
        )
        text = storage.source_path("src").read_text(encoding="utf-8")
        # ensure_ascii=False is used so the accent is stored verbatim
        assert "hipertensión" in text


class TestExistsAndClear:
    def test_exists_when_non_empty(self, storage):
        assert storage.exists("src") is False
        storage.save(Document("d1", "t", "u"), "src")
        assert storage.exists("src") is True

    def test_clear_removes_file(self, storage):
        storage.save(Document("d1", "t", "u"), "src")
        storage.clear("src")
        assert storage.exists("src") is False
