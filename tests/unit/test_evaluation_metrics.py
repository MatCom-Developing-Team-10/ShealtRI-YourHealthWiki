"""Unit tests for ``modules.evaluation.metrics``.

The metrics are pure functions, so these tests are deterministic and fast.
Each test pins a hand-computed expected value from the lecture's
confusion-matrix notation so a silent change in any formula trips here.
"""

from __future__ import annotations

import math

import pytest

from modules.evaluation.metrics import (
    average_precision,
    confusion_counts,
    dcg_at_k,
    f1,
    f_measure,
    ndcg_at_k,
    precision,
    precision_at_k,
    recall,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


class TestConfusionCounts:
    def test_basic_split(self):
        retrieved = ["a", "b", "c", "d"]
        relevance = {"a": 1, "c": 1, "e": 1}  # 3 relevant; a,c retrieved
        c = confusion_counts(retrieved, relevance)
        assert c.rr == 2          # a, c
        assert c.ri == 2          # b, d
        assert c.nr == 1          # e
        assert c.ni == 0          # corpus_size not given

    def test_corpus_size_computes_ni(self):
        retrieved = ["a", "b"]
        relevance = {"a": 1, "c": 1}
        c = confusion_counts(retrieved, relevance, corpus_size=10)
        # rr=1 (a), ri=1 (b), nr=1 (c), so ni = 10 - 3 = 7
        assert c.ni == 7

    def test_duplicates_collapsed(self):
        c = confusion_counts(["a", "a", "a"], {"a": 1})
        assert c.rr == 1
        assert c.ri == 0


# ---------------------------------------------------------------------------
# Precision / recall / F-measure
# ---------------------------------------------------------------------------


class TestPrecisionRecall:
    def test_precision_basic(self):
        assert precision(["a", "b"], {"a": 1}) == pytest.approx(0.5)

    def test_recall_basic(self):
        assert recall(["a"], {"a": 1, "b": 1}) == pytest.approx(0.5)

    def test_precision_empty_retrieved(self):
        # Defined as 0 in the implementation (lecture says undefined)
        assert precision([], {"a": 1}) == 0.0

    def test_recall_no_relevant(self):
        assert recall(["a"], {}) == 0.0

    def test_grade_zero_is_not_relevant(self):
        # grade 0 should be treated as not relevant
        assert precision(["a"], {"a": 0}) == 0.0


class TestFMeasure:
    def test_f1_balanced(self):
        # P=0.5, R=0.5 → F1 = 0.5
        assert f1(0.5, 0.5) == pytest.approx(0.5)

    def test_f1_both_zero(self):
        assert f1(0.0, 0.0) == 0.0

    def test_f1_one_zero(self):
        assert f1(0.0, 1.0) == 0.0
        assert f1(1.0, 0.0) == 0.0

    def test_f_measure_beta_prefers_precision(self):
        # beta < 1 → precision weighs more
        # With P=1.0 and R=0.5, F should be closer to P
        f_high_p = f_measure(1.0, 0.5, beta=0.5)
        f_high_r = f_measure(0.5, 1.0, beta=0.5)
        assert f_high_p > f_high_r

    def test_f_measure_beta_prefers_recall(self):
        # beta > 1 → recall weighs more
        f_high_p = f_measure(1.0, 0.5, beta=2.0)
        f_high_r = f_measure(0.5, 1.0, beta=2.0)
        assert f_high_r > f_high_p


# ---------------------------------------------------------------------------
# @k metrics
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_top_k_hits(self):
        # Top-3 contains 'a','b','c'. Only a,b are relevant.
        assert precision_at_k(["a", "b", "c", "d"], {"a": 1, "b": 1}, k=3) == pytest.approx(2 / 3)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a"], {"a": 1}, k=0) == 0.0

    def test_k_larger_than_retrieved(self):
        # Only 2 docs retrieved, asking for k=5 → uses what we have (top-2)
        assert precision_at_k(["a", "b"], {"a": 1}, k=5) == pytest.approx(0.5)


class TestRecallAtK:
    def test_top_k_hits(self):
        assert recall_at_k(["a", "b", "c"], {"a": 1, "b": 1, "d": 1}, k=3) == pytest.approx(2 / 3)

    def test_no_relevant_returns_zero(self):
        assert recall_at_k(["a"], {}, k=5) == 0.0


# ---------------------------------------------------------------------------
# Reciprocal rank / MAP
# ---------------------------------------------------------------------------


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a": 1}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["b", "a"], {"a": 1}) == 0.5

    def test_no_relevant(self):
        assert reciprocal_rank(["a", "b"], {"c": 1}) == 0.0


class TestAveragePrecision:
    def test_perfect_ranking(self):
        # All relevant items at the top: AP = 1.0
        assert average_precision(["a", "b"], {"a": 1, "b": 1}) == pytest.approx(1.0)

    def test_known_textbook_example(self):
        # 5 docs, relevant at positions 1, 3, 5
        # AP = (1/3) * (1/1 + 2/3 + 3/5) = (1/3)*(2.2667) ≈ 0.7556
        ranked = ["a", "x", "b", "y", "c"]
        relevance = {"a": 1, "b": 1, "c": 1}
        expected = (1 / 3) * (1.0 + (2 / 3) + (3 / 5))
        assert average_precision(ranked, relevance) == pytest.approx(expected)

    def test_no_relevant_returns_zero(self):
        assert average_precision(["a"], {}) == 0.0


# ---------------------------------------------------------------------------
# DCG / NDCG
# ---------------------------------------------------------------------------


class TestDCG:
    def test_dcg_position_one_no_discount(self):
        # rel=2 at position 1: DCG = 2 / log2(2) = 2
        assert dcg_at_k(["a"], {"a": 2}, k=1) == pytest.approx(2.0)

    def test_dcg_zero_relevance_contributes_nothing(self):
        assert dcg_at_k(["a"], {"a": 0}, k=1) == 0.0

    def test_dcg_graded_relevance(self):
        # Position 1: rel=3 → 3 / log2(2) = 3
        # Position 2: rel=2 → 2 / log2(3)
        # Position 3: rel=1 → 1 / log2(4) = 0.5
        relevance = {"a": 3, "b": 2, "c": 1}
        expected = 3.0 + 2 / math.log2(3) + 1 / math.log2(4)
        assert dcg_at_k(["a", "b", "c"], relevance, k=3) == pytest.approx(expected)


class TestNDCG:
    def test_perfect_ranking_yields_one(self):
        # System returns relevant docs in descending order of grade
        assert ndcg_at_k(["a", "b", "c"], {"a": 3, "b": 2, "c": 1}, k=3) == pytest.approx(1.0)

    def test_worst_ranking_below_one(self):
        # Reverse: less relevant first
        assert ndcg_at_k(["c", "b", "a"], {"a": 3, "b": 2, "c": 1}, k=3) < 1.0

    def test_ndcg_zero_when_no_relevant_in_top_k(self):
        assert ndcg_at_k(["x", "y"], {"a": 1}, k=2) == 0.0

    def test_ndcg_no_relevant_at_all(self):
        assert ndcg_at_k(["x"], {}, k=5) == 0.0
