"""Unit tests for the feedback stores (in-memory + JSONL)."""

from __future__ import annotations

import json

import pytest

from plugins.feedback.models import RelevanceJudgment
from plugins.feedback.store import InMemoryFeedbackStore, JSONLFeedbackStore


# Both store implementations honour the same contract, so the same suite
# runs against both via parametrisation.


def _store_factories(tmp_path):
    return [
        ("InMemoryFeedbackStore", lambda: InMemoryFeedbackStore()),
        (
            "JSONLFeedbackStore",
            lambda: JSONLFeedbackStore(tmp_path / "feedback.jsonl"),
        ),
    ]


@pytest.fixture(params=["memory", "jsonl"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryFeedbackStore()
    return JSONLFeedbackStore(tmp_path / "feedback.jsonl")


class TestAddAndGet:
    def test_empty_store_returns_empty_tuples(self, store):
        assert store.get_for_query("anything") == ([], [])

    def test_add_relevant_only(self, store):
        store.add(RelevanceJudgment("diabetes", "d1", relevant=True))
        rel, non_rel = store.get_for_query("diabetes")
        assert rel == ["d1"]
        assert non_rel == []

    def test_add_non_relevant_only(self, store):
        store.add(RelevanceJudgment("diabetes", "d_bad", relevant=False))
        rel, non_rel = store.get_for_query("diabetes")
        assert rel == []
        assert non_rel == ["d_bad"]

    def test_mixed_judgments_for_same_query(self, store):
        store.add(RelevanceJudgment("diabetes", "d1", relevant=True))
        store.add(RelevanceJudgment("diabetes", "d2", relevant=False))
        store.add(RelevanceJudgment("diabetes", "d3", relevant=True))
        rel, non_rel = store.get_for_query("diabetes")
        assert set(rel) == {"d1", "d3"}
        assert non_rel == ["d2"]

    def test_judgments_partitioned_by_query(self, store):
        store.add(RelevanceJudgment("diabetes", "d1", relevant=True))
        store.add(RelevanceJudgment("asma", "d2", relevant=True))
        rel_d, _ = store.get_for_query("diabetes")
        rel_a, _ = store.get_for_query("asma")
        assert rel_d == ["d1"]
        assert rel_a == ["d2"]


class TestQueryNormalisation:
    def test_case_insensitive_lookup(self, store):
        store.add(RelevanceJudgment("Diabetes", "d1", relevant=True))
        rel, _ = store.get_for_query("DIABETES")
        assert rel == ["d1"]

    def test_whitespace_insensitive(self, store):
        store.add(RelevanceJudgment("  diabetes ", "d1", relevant=True))
        rel, _ = store.get_for_query("diabetes")
        assert rel == ["d1"]


class TestOverride:
    def test_latest_judgment_for_same_pair_wins(self, store):
        # First mark relevant, then change mind
        store.add(RelevanceJudgment("diabetes", "d1", relevant=True))
        store.add(RelevanceJudgment("diabetes", "d1", relevant=False))
        rel, non_rel = store.get_for_query("diabetes")
        assert rel == []
        assert non_rel == ["d1"]


class TestClear:
    def test_clear_empties_store(self, store):
        store.add(RelevanceJudgment("diabetes", "d1", relevant=True))
        store.clear()
        assert len(store) == 0
        assert store.get_for_query("diabetes") == ([], [])


class TestLen:
    def test_len_reflects_records_added(self, store):
        assert len(store) == 0
        store.add(RelevanceJudgment("q", "d1", relevant=True))
        store.add(RelevanceJudgment("q", "d2", relevant=False))
        assert len(store) == 2


# ---------------------------------------------------------------------------
# JSONL-specific tests
# ---------------------------------------------------------------------------


class TestJSONLPersistence:
    def test_records_survive_new_instance(self, tmp_path):
        path = tmp_path / "f.jsonl"
        store_a = JSONLFeedbackStore(path)
        store_a.add(RelevanceJudgment("diabetes", "d1", relevant=True))
        store_a.add(RelevanceJudgment("diabetes", "d2", relevant=False))

        # Re-open with a fresh instance
        store_b = JSONLFeedbackStore(path)
        rel, non_rel = store_b.get_for_query("diabetes")
        assert rel == ["d1"]
        assert non_rel == ["d2"]

    def test_file_format_is_jsonl(self, tmp_path):
        path = tmp_path / "f.jsonl"
        store = JSONLFeedbackStore(path)
        store.add(RelevanceJudgment("q", "d1", relevant=True))
        store.add(RelevanceJudgment("q", "d2", relevant=False))
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert set(record.keys()) == {"query_text", "doc_id", "relevant", "timestamp"}

    def test_creates_missing_parent_directory(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "path" / "f.jsonl"
        store = JSONLFeedbackStore(path)
        store.add(RelevanceJudgment("q", "d1", relevant=True))
        assert path.exists()

    def test_malformed_line_skipped(self, tmp_path):
        path = tmp_path / "f.jsonl"
        store = JSONLFeedbackStore(path)
        store.add(RelevanceJudgment("q", "d1", relevant=True))
        # Inject a bad line in the middle
        with path.open("a", encoding="utf-8") as f:
            f.write("not json at all\n")
        store.add(RelevanceJudgment("q", "d2", relevant=False))

        rel, non_rel = store.get_for_query("q")
        assert rel == ["d1"]
        assert non_rel == ["d2"]
        # __len__ should also count only valid records
        assert len(store) == 2

    def test_clear_truncates_file(self, tmp_path):
        path = tmp_path / "f.jsonl"
        store = JSONLFeedbackStore(path)
        store.add(RelevanceJudgment("q", "d1", relevant=True))
        store.clear()
        assert path.read_text(encoding="utf-8") == ""
