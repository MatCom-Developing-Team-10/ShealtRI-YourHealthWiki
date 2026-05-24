"""Regression test for on-disk artifact format compatibility.

Guards against silent serialisation changes in:
    - IndexStore (corpus.joblib, manifest.json, doc_ids.txt, spell_vocab.txt)
    - TfidfProcessor (tfidf.joblib: vocabulary + idf + n_docs)
    - LSIModel (svd.joblib)
    - FileSystemDocumentStore (one JSON file per doc)
    - RawDocumentStorage (JSONL line records)

A round-trip on every artifact validates that the format we wrote yesterday
can still be read today. Schema version is asserted explicitly for the
IndexStore manifest so a bump to v2.0 fails this test until the migration
path is in place.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from core.interfaces import IndexedCorpus
from core.models import Document
from infra.storage import RawDocumentStorage
from modules.indexer.document_store import FileSystemDocumentStore
from modules.indexer.index_store import IndexStore
from modules.retriever.lsi_model import LSIModel
from modules.retriever.tfidf_processor import TfidfProcessor
from modules.text_processor.spell_checker import TrieSpellChecker


# ---------------------------------------------------------------------------
# IndexStore: artefact layout & manifest schema
# ---------------------------------------------------------------------------


def _tiny_corpus():
    docs = [
        Document("d1", "alpha beta", "u1"),
        Document("d2", "beta gamma", "u2"),
    ]
    return IndexedCorpus(
        documents=docs,
        processed_texts=["alpha beta", "beta gamma"],
        inverted_index={
            "alpha": [(0, 1)],
            "beta": [(0, 1), (1, 1)],
            "gamma": [(1, 1)],
        },
        vocabulary=["alpha", "beta", "gamma"],
    )


class TestIndexStoreFormat:
    def test_expected_files_after_save(self, tmp_path):
        store = IndexStore(storage_dir=tmp_path / "idx")
        store.save(_tiny_corpus())
        for name in ("corpus.joblib", "doc_ids.txt", "manifest.json"):
            assert (tmp_path / "idx" / name).exists(), f"missing {name}"

    def test_manifest_schema_v1(self, tmp_path):
        store = IndexStore(storage_dir=tmp_path / "idx")
        store.save(_tiny_corpus())
        manifest = json.loads(
            (tmp_path / "idx" / "manifest.json").read_text(encoding="utf-8")
        )
        # Hard pin the schema. When you bump to v2.0, add a migration test
        # and update the assertion below in the same commit.
        assert manifest["schema_version"] == "1.0"
        for required in ("created_at", "updated_at", "n_documents", "n_terms"):
            assert required in manifest, f"manifest missing {required}"

    def test_doc_ids_sidecar_one_per_line(self, tmp_path):
        store = IndexStore(storage_dir=tmp_path / "idx")
        corpus = _tiny_corpus()
        store.save(corpus)
        content = (tmp_path / "idx" / "doc_ids.txt").read_text(encoding="utf-8")
        lines = [l for l in content.splitlines() if l]
        assert lines == [d.doc_id for d in corpus.documents]

    def test_spell_vocab_sorted_one_per_line(self, tmp_path):
        store = IndexStore(storage_dir=tmp_path / "idx")
        checker = TrieSpellChecker()
        for w in ("zeta", "alpha", "mu"):
            checker._insert(w)
        store.save_spell_vocabulary(checker)
        content = (tmp_path / "idx" / "spell_vocab.txt").read_text(encoding="utf-8")
        lines = [l for l in content.splitlines() if l]
        # Order is fixed (sorted) — guarantees stable diffs across saves
        assert lines == sorted(lines)
        assert set(lines) == {"alpha", "mu", "zeta"}


# ---------------------------------------------------------------------------
# TfidfProcessor: joblib payload shape
# ---------------------------------------------------------------------------


class TestTfidfFormat:
    def test_joblib_payload_keys(self, tmp_path):
        tfidf = TfidfProcessor()
        tfidf.fit(_tiny_corpus())
        tfidf.save(tmp_path)
        payload = joblib.load(tmp_path / "tfidf.joblib")
        # Three keys, in this order historically — protect against drift.
        assert set(payload.keys()) == {"vocabulary", "idf", "n_docs"}
        assert isinstance(payload["vocabulary"], list)
        assert isinstance(payload["idf"], np.ndarray)
        assert isinstance(payload["n_docs"], int)


# ---------------------------------------------------------------------------
# LSIModel: SVD object survives save/load
# ---------------------------------------------------------------------------


class TestLsiModelFormat:
    def test_svd_round_trip(self, tmp_path):
        rng = np.random.default_rng(0)
        m = csr_matrix(rng.random((4, 6), dtype=np.float32))
        model = LSIModel(n_components=2)
        original_doc_vectors = model.fit(m)
        model.save(tmp_path)
        loaded = LSIModel.load(tmp_path)

        # The loaded model must project the same matrix to the same latent space.
        q = csr_matrix(m.toarray()[0:1])
        np.testing.assert_allclose(
            loaded.project_query(q), model.project_query(q), rtol=1e-5
        )


# ---------------------------------------------------------------------------
# FileSystemDocumentStore: JSON-per-document shape
# ---------------------------------------------------------------------------


class TestDocumentStoreFormat:
    def test_per_document_json_keys(self, tmp_path):
        store = FileSystemDocumentStore(storage_dir=str(tmp_path / "store"))
        store.add_documents(
            [Document("d1", "text", "https://x", metadata={"k": 1})]
        )
        files = list((tmp_path / "store").glob("*.json"))
        assert len(files) == 1
        record = json.loads(files[0].read_text(encoding="utf-8"))
        assert set(record.keys()) == {"doc_id", "text", "url", "metadata"}
        assert record["metadata"] == {"k": 1}


# ---------------------------------------------------------------------------
# RawDocumentStorage: JSONL record shape
# ---------------------------------------------------------------------------


class TestRawStorageFormat:
    def test_jsonl_record_shape(self, tmp_path):
        storage = RawDocumentStorage(str(tmp_path / "raw"))
        storage.save(
            Document("d1", "texto", "https://x", metadata={"title": "T"}),
            "src",
        )
        path = storage.source_path("src")
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert set(record.keys()) == {"doc_id", "text", "url", "metadata"}
        assert record["doc_id"] == "d1"
        assert record["metadata"]["title"] == "T"
