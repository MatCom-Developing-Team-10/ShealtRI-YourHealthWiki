"""Tests for DocumentLoader.

LangChain is an optional dependency. If it's not installed, the module is
skipped via ``importorskip``.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_community")

from modules.document_loader import DocumentLoader, DocumentLoaderError


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


class TestLoadFromJson:
    def test_load_single_document(self, loader, tmp_path):
        data = {"doc_id": "d1", "text": "hello", "url": "https://x"}
        path = tmp_path / "doc.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        docs = loader.load_from_json(path)
        assert len(docs) == 1
        assert docs[0].doc_id == "d1"

    def test_load_list_of_documents(self, loader, tmp_path):
        data = [
            {"doc_id": "d1", "text": "t1", "url": "u1"},
            {"doc_id": "d2", "text": "t2", "url": "u2"},
        ]
        path = tmp_path / "docs.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        docs = loader.load_from_json(path)
        assert len(docs) == 2
        assert {d.doc_id for d in docs} == {"d1", "d2"}

    def test_invalid_json_raises(self, loader, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("not valid {", encoding="utf-8")
        with pytest.raises(DocumentLoaderError):
            loader.load_from_json(path)

    def test_missing_required_field_raises(self, loader, tmp_path):
        data = [{"doc_id": "d1", "text": "t"}]  # missing url
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(DocumentLoaderError):
            loader.load_from_json(path)

    def test_metadata_passes_through(self, loader, tmp_path):
        data = {
            "doc_id": "d1",
            "text": "t",
            "url": "u",
            "metadata": {"title": "T"},
        }
        path = tmp_path / "doc.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        docs = loader.load_from_json(path)
        assert docs[0].metadata == {"title": "T"}


class TestLoadFromList:
    def test_basic(self, loader):
        out = loader.load_from_list(
            [{"doc_id": "d1", "text": "t", "url": "u"}]
        )
        assert len(out) == 1

    def test_invalid_entry_raises(self, loader):
        with pytest.raises(DocumentLoaderError):
            loader.load_from_list([{"doc_id": "d1"}])  # missing text & url


class TestLoadFromDirectory:
    def test_directory_not_found(self, loader, tmp_path):
        with pytest.raises(DocumentLoaderError):
            loader.load_from_directory(tmp_path / "nope")

    def test_path_is_file_raises(self, loader, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hi", encoding="utf-8")
        with pytest.raises(DocumentLoaderError):
            loader.load_from_directory(f)

    def test_loads_json_files(self, loader, tmp_path):
        (tmp_path / "a.json").write_text(
            json.dumps({"doc_id": "d1", "text": "t1", "url": "u1"}),
            encoding="utf-8",
        )
        (tmp_path / "b.json").write_text(
            json.dumps({"doc_id": "d2", "text": "t2", "url": "u2"}),
            encoding="utf-8",
        )
        docs = loader.load_from_directory(tmp_path)
        assert len(docs) == 2
        assert {d.doc_id for d in docs} == {"d1", "d2"}
