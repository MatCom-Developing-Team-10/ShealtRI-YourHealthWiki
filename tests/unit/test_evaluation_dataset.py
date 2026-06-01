"""Tests for ``modules.evaluation.dataset.load_dataset``."""

from __future__ import annotations

import json

import pytest

from modules.evaluation.dataset import load_dataset
from modules.evaluation.models import EvalQuery, EvaluationDataset


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TestLoadDataset:
    def test_round_trip(self, tmp_path):
        queries_path = tmp_path / "queries.json"
        qrels_path = tmp_path / "qrels.json"
        _write(
            queries_path,
            [
                {"query_id": "q1", "text": "diabetes"},
                {"query_id": "q2", "text": "asma"},
            ],
        )
        _write(qrels_path, {"q1": {"d1": 2, "d2": 1}, "q2": {"d3": 1}})

        ds = load_dataset(queries_path, qrels_path)
        assert isinstance(ds, EvaluationDataset)
        assert len(ds) == 2
        assert ds.queries[0] == EvalQuery(query_id="q1", text="diabetes")
        assert ds.qrels["q1"] == {"d1": 2, "d2": 1}

    def test_missing_queries_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "missing.json", tmp_path / "irrelevant.json")

    def test_missing_qrels_file_raises(self, tmp_path):
        queries_path = tmp_path / "queries.json"
        _write(queries_path, [{"query_id": "q1", "text": "x"}])
        with pytest.raises(FileNotFoundError):
            load_dataset(queries_path, tmp_path / "missing-qrels.json")

    def test_query_missing_required_field_raises(self, tmp_path):
        queries_path = tmp_path / "queries.json"
        qrels_path = tmp_path / "qrels.json"
        _write(queries_path, [{"query_id": "q1"}])  # no 'text'
        _write(qrels_path, {})
        with pytest.raises(ValueError, match="missing"):
            load_dataset(queries_path, qrels_path)

    def test_qrels_referencing_unknown_query_raises(self, tmp_path):
        queries_path = tmp_path / "queries.json"
        qrels_path = tmp_path / "qrels.json"
        _write(queries_path, [{"query_id": "q1", "text": "x"}])
        _write(qrels_path, {"q_unknown": {"d1": 1}})
        with pytest.raises(ValueError, match="unknown query_id"):
            load_dataset(queries_path, qrels_path)


class TestDatasetHelpers:
    def test_relevance_for_known_query(self):
        ds = EvaluationDataset(
            queries=[EvalQuery("q1", "text")], qrels={"q1": {"d1": 1}}
        )
        assert ds.relevance_for("q1") == {"d1": 1}

    def test_relevance_for_unknown_query(self):
        ds = EvaluationDataset(queries=[EvalQuery("q1", "x")], qrels={})
        assert ds.relevance_for("q1") == {}
