"""Unit tests for ``plugins.feedback.models``."""

from __future__ import annotations

from plugins.feedback.models import RelevanceJudgment


class TestRelevanceJudgment:
    def test_construction_assigns_timestamp(self):
        j = RelevanceJudgment(query_text="diabetes", doc_id="d1", relevant=True)
        assert j.timestamp  # auto-filled
        assert j.query_text == "diabetes"
        assert j.doc_id == "d1"
        assert j.relevant is True

    def test_timestamp_explicit_override(self):
        ts = "2026-01-01T00:00:00+00:00"
        j = RelevanceJudgment(
            query_text="q", doc_id="d", relevant=False, timestamp=ts
        )
        assert j.timestamp == ts


class TestNormaliseQuery:
    def test_lowercases(self):
        assert RelevanceJudgment.normalise_query("DIABETES") == "diabetes"

    def test_strips_whitespace(self):
        assert RelevanceJudgment.normalise_query("  diabetes  ") == "diabetes"

    def test_collapses_internal_whitespace(self):
        assert (
            RelevanceJudgment.normalise_query("hipertensión   arterial")
            == "hipertensión arterial"
        )

    def test_idempotent(self):
        for s in ("diabetes", "hipertensión arterial", "  asma  "):
            once = RelevanceJudgment.normalise_query(s)
            twice = RelevanceJudgment.normalise_query(once)
            assert once == twice
