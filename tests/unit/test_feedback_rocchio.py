"""Unit tests for ``plugins.feedback.rocchio.RocchioReweighter``."""

from __future__ import annotations

import pytest

from plugins.feedback.rocchio import RocchioReweighter


class TestRocchioReweight:
    def test_no_feedback_returns_copy_not_alias(self):
        rocchio = RocchioReweighter()
        original = [1.0, 2.0, 3.0]
        out = rocchio.reweight(original, [], [])
        assert out == original
        # Must not share identity — modifying the original must not leak.
        original[0] = 999.0
        assert out[0] == 1.0

    def test_textbook_formula(self):
        # α=1, β=0.75, γ=0.15
        rocchio = RocchioReweighter()
        q = [1.0, 0.0, 0.0]
        rel = [[1.0, 1.0, 0.0]]
        nonrel = [[0.0, 0.0, 1.0]]
        out = rocchio.reweight(q, rel, nonrel)
        # q' = 1·q + 0.75·(1,1,0) − 0.15·(0,0,1)
        assert out[0] == pytest.approx(1.0 + 0.75)
        assert out[1] == pytest.approx(0.0 + 0.75)
        assert out[2] == pytest.approx(0.0 - 0.15)

    def test_pulls_toward_relevant_centroid(self):
        rocchio = RocchioReweighter(alpha=1.0, beta=1.0, gamma=0.0)
        q = [0.0, 0.0]
        rel = [[2.0, 0.0], [4.0, 0.0]]  # centroid = (3, 0)
        out = rocchio.reweight(q, rel, [])
        assert out == [pytest.approx(3.0), pytest.approx(0.0)]

    def test_pushes_away_from_non_relevant_centroid(self):
        rocchio = RocchioReweighter(alpha=1.0, beta=0.0, gamma=1.0)
        q = [0.0, 0.0]
        nonrel = [[1.0, 0.0], [3.0, 0.0]]  # centroid = (2, 0)
        out = rocchio.reweight(q, [], nonrel)
        assert out == [pytest.approx(-2.0), pytest.approx(0.0)]

    def test_centroid_is_arithmetic_mean(self):
        rocchio = RocchioReweighter(alpha=0.0, beta=1.0, gamma=0.0)
        q = [0.0, 0.0]
        rel = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  # mean = (2/3, 2/3)
        out = rocchio.reweight(q, rel, [])
        assert out[0] == pytest.approx(2 / 3)
        assert out[1] == pytest.approx(2 / 3)

    def test_alpha_controls_anchor_strength(self):
        weak = RocchioReweighter(alpha=0.1, beta=1.0, gamma=0.0)
        strong = RocchioReweighter(alpha=10.0, beta=1.0, gamma=0.0)
        q = [1.0]
        rel = [[5.0]]
        # weak: 0.1*1 + 1*5 = 5.1; strong: 10*1 + 1*5 = 15
        assert weak.reweight(q, rel, [])[0] == pytest.approx(5.1)
        assert strong.reweight(q, rel, [])[0] == pytest.approx(15.0)


class TestErrorHandling:
    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            RocchioReweighter().reweight([], [], [])

    def test_dimension_mismatch_relevant_raises(self):
        with pytest.raises(ValueError, match="dimension 4, expected 3"):
            RocchioReweighter().reweight([1.0, 2.0, 3.0], [[1.0, 1.0, 1.0, 1.0]], [])

    def test_dimension_mismatch_non_relevant_raises(self):
        with pytest.raises(ValueError, match="dimension 2, expected 3"):
            RocchioReweighter().reweight([1.0, 2.0, 3.0], [], [[1.0, 1.0]])
