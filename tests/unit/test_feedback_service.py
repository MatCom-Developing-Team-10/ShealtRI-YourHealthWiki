"""Unit tests for ``plugins.feedback.service.RelevanceFeedbackService``."""

from __future__ import annotations

import pytest

from plugins.feedback.rocchio import RocchioReweighter
from plugins.feedback.service import RelevanceFeedbackService
from plugins.feedback.store import InMemoryFeedbackStore


def _lookup(table: dict[str, list[float]]):
    """Build an ``embedding_lookup`` callable from a dict."""
    return table.get


# ---------------------------------------------------------------------------
# Recording feedback
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_persists_to_store(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record("diabetes", "d1", relevant=True)
        assert len(service.store) == 1

    def test_record_many_handles_both_sides(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record_many(
            "diabetes",
            relevant_ids=["d1", "d2"],
            non_relevant_ids=["d_bad"],
        )
        assert len(service.store) == 3

    def test_record_many_none_inputs_is_noop(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record_many("diabetes")
        assert len(service.store) == 0


# ---------------------------------------------------------------------------
# Querying state
# ---------------------------------------------------------------------------


class TestHasFeedback:
    def test_false_when_no_judgments(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        assert service.has_feedback("diabetes") is False

    def test_true_after_recording(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record("diabetes", "d1", relevant=True)
        assert service.has_feedback("diabetes") is True

    def test_partitioned_by_query(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record("diabetes", "d1", relevant=True)
        assert service.has_feedback("asma") is False


# ---------------------------------------------------------------------------
# Applying feedback
# ---------------------------------------------------------------------------


class TestApplyToQuery:
    def test_no_feedback_returns_copy_unchanged(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        original = [1.0, 2.0, 3.0]
        out = service.apply_to_query("diabetes", original, lambda d: None)
        assert out == original
        # Different object — must not be the same reference
        original[0] = 999.0
        assert out[0] == 1.0

    def test_rocchio_applied_when_relevant_known(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record("diabetes", "d1", relevant=True)

        # alpha=1, beta=0.75, gamma=0.15 (defaults)
        q = [1.0, 0.0]
        out = service.apply_to_query(
            "diabetes", q, _lookup({"d1": [0.0, 1.0]}),
        )
        # q' = 1*(1,0) + 0.75*(0,1) - 0.15*(0,0) = (1.0, 0.75)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.75)

    def test_rocchio_applied_with_only_non_relevant(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record("diabetes", "d_bad", relevant=False)

        q = [1.0, 1.0]
        out = service.apply_to_query(
            "diabetes", q, _lookup({"d_bad": [1.0, 0.0]}),
        )
        # q' = 1*(1,1) + 0.75*(0,0) - 0.15*(1,0) = (0.85, 1.0)
        assert out[0] == pytest.approx(0.85)
        assert out[1] == pytest.approx(1.0)

    def test_unknown_embeddings_falls_back_to_original_query(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record("diabetes", "d_missing", relevant=True)
        original = [1.0, 2.0, 3.0]
        out = service.apply_to_query(
            "diabetes", original, lambda d: None,
        )
        assert out == original

    def test_query_text_normalised_when_applying(self):
        service = RelevanceFeedbackService(InMemoryFeedbackStore())
        service.record("Diabetes", "d1", relevant=True)
        out = service.apply_to_query(
            "  DIABETES  ", [1.0, 0.0], _lookup({"d1": [0.0, 1.0]}),
        )
        assert out != [1.0, 0.0]  # Rocchio kicked in


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_store_is_in_memory(self):
        service = RelevanceFeedbackService()
        assert isinstance(service.store, InMemoryFeedbackStore)

    def test_default_rocchio_is_textbook(self):
        service = RelevanceFeedbackService()
        assert service.rocchio.alpha == 1.0
        assert service.rocchio.beta == 0.75
        assert service.rocchio.gamma == 0.15

    def test_custom_rocchio_honoured(self):
        custom = RocchioReweighter(alpha=2.0, beta=0.5, gamma=0.5)
        service = RelevanceFeedbackService(rocchio=custom)
        assert service.rocchio is custom
