"""Extra coverage for ``modules.indexer.document_store.FileSystemDocumentStore``.

The base test suite covers the happy path and basic error handling; this
file fills in the corners: ``list_all_ids``, batch reads with corrupted
files, and delete on a locked file.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.models import Document
from modules.indexer.document_store import (
    DocumentReadError,
    DocumentStoreError,
    FileSystemDocumentStore,
)


@pytest.fixture
def store(tmp_path) -> FileSystemDocumentStore:
    return FileSystemDocumentStore(storage_dir=str(tmp_path / "store"))


class TestListAllIds:
    def test_empty_directory_returns_empty(self, store):
        assert store.list_all_ids() == []

    def test_returns_added_doc_ids(self, store):
        store.add_documents(
            [
                Document("d1", "t1", "u1"),
                Document("d2", "t2", "u2"),
                Document("d3", "t3", "u3"),
            ]
        )
        assert sorted(store.list_all_ids()) == ["d1", "d2", "d3"]

    def test_preserves_original_id_for_hashed_filenames(self, store):
        # An ID with special chars is hashed for the filename — but the
        # original ID is written inside the JSON body and recovered by
        # list_all_ids().
        original = "weird/id*with?chars"
        store.add_documents([Document(original, "text", "url")])
        ids = store.list_all_ids()
        assert ids == [original]

    def test_skips_corrupted_json_files(self, store, tmp_path):
        store.add_documents([Document("d1", "t", "u")])
        # Drop a malformed JSON file alongside the good one
        bad = tmp_path / "store" / "broken.json"
        bad.write_text("not json", encoding="utf-8")
        ids = store.list_all_ids()
        assert ids == ["d1"]

    def test_skips_json_missing_doc_id_field(self, store, tmp_path):
        import json

        store.add_documents([Document("good", "t", "u")])
        (tmp_path / "store" / "no_id.json").write_text(
            json.dumps({"text": "hi"}), encoding="utf-8"
        )
        ids = store.list_all_ids()
        assert ids == ["good"]


class TestGetByIdsErrorHandling:
    def test_unreadable_document_skipped_not_raised(self, store, tmp_path):
        # Two good docs + one whose path triggers a DocumentReadError
        store.add_documents(
            [Document("good1", "t", "u"), Document("good2", "t", "u")]
        )

        with patch.object(
            store, "get_by_id", side_effect=[
                DocumentReadError("corrupt"),  # for 'bad'
                Document("good1", "t", "u"),
                Document("good2", "t", "u"),
            ]
        ):
            out = store.get_by_ids(["bad", "good1", "good2"])
        assert [d.doc_id for d in out] == ["good1", "good2"]


class TestDeleteErrorPath:
    def test_delete_raises_document_store_error_on_oserror(self, store):
        store.add_documents([Document("d1", "t", "u")])
        with patch(
            "modules.indexer.document_store.Path.unlink",
            side_effect=OSError("file in use"),
        ):
            with pytest.raises(DocumentStoreError, match="Cannot delete"):
                store.delete("d1")


class TestPathSanitizationEdgeCases:
    def test_dot_id_is_hashed(self, store, tmp_path):
        # A doc_id consisting of '.' or '..' must not produce a hidden file
        store.add_documents([Document(".", "t", "u")])
        files = list((tmp_path / "store").glob("*.json"))
        assert len(files) == 1
        assert not files[0].name.startswith(".")

    def test_hidden_file_id_is_hashed(self, store, tmp_path):
        # A doc_id starting with '.' would create a hidden file on POSIX — must hash
        store.add_documents([Document(".secret_doc", "t", "u")])
        files = list((tmp_path / "store").glob("*.json"))
        assert len(files) == 1
        assert not files[0].name.startswith(".")
