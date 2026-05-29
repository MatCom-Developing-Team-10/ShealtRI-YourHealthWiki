"""Tests for ``modules.evaluation.service.EvaluationService``.

The service is decoupled from any concrete retriever via the ``search_fn``
callable, so these tests inject a stub function returning deterministic
result lists.
"""

from __future__ import annotations

import pytest

from modules.evaluation.models import (
    EvalQuery,
    EvaluationDataset,
    EvaluationReport,
    QueryEvalResult,
)
from modules.evaluation.service import EvaluationService


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        queries=[
            EvalQuery("q1", "diabetes"),
            EvalQuery("q2", "asma"),
        ],
        qrels={
            "q1": {"d1": 1, "d2": 1, "d3": 1},
            "q2": {"d4": 1},
        },
    )


def _make_search_fn(results_by_query: dict[str, list[str]]):
    def search(query_text: str, top_k: int):
        # Find the corresponding query_id in the dataset by text
        for qid, text in [("q1", "diabetes"), ("q2", "asma")]:
            if text == query_text:
                return results_by_query.get(qid, [])[:top_k]
        return []
    return search


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_report_for_each_query(self):
        search_fn = _make_search_fn(
            {"q1": ["d1", "d2", "x"], "q2": ["d4"]}
        )
        report = EvaluationService(search_fn, k=3).evaluate(_dataset())
        assert isinstance(report, EvaluationReport)
        assert len(report.per_query) == 2
        assert report.k == 3

    def test_per_query_metrics_present(self):
        search_fn = _make_search_fn({"q1": ["d1"], "q2": ["d4"]})
        report = EvaluationService(search_fn, k=5).evaluate(_dataset())
        for result in report.per_query:
            assert {"P@5", "R@5", "F1@5", "NDCG@5", "AP", "RR"} <= result.metrics.keys()

    def test_q1_perfect_p_at_k_when_all_top_relevant(self):
        # q1 has 3 relevant docs; retrieve them all at the top
        search_fn = _make_search_fn({"q1": ["d1", "d2", "d3"], "q2": []})
        report = EvaluationService(search_fn, k=3).evaluate(_dataset())
        q1 = next(r for r in report.per_query if r.query_id == "q1")
        assert q1.metrics["P@3"] == pytest.approx(1.0)
        assert q1.metrics["R@3"] == pytest.approx(1.0)
        assert q1.metrics["AP"] == pytest.approx(1.0)
        assert q1.metrics["RR"] == pytest.approx(1.0)

    def test_num_relevant_and_num_retrieved_recorded(self):
        search_fn = _make_search_fn({"q1": ["d1", "x"], "q2": []})
        report = EvaluationService(search_fn, k=10).evaluate(_dataset())
        q1 = next(r for r in report.per_query if r.query_id == "q1")
        assert q1.num_relevant == 3
        assert q1.num_retrieved == 2


class TestAggregate:
    def test_aggregated_renames_ap_to_map_and_rr_to_mrr(self):
        search_fn = _make_search_fn({"q1": ["d1"], "q2": ["d4"]})
        report = EvaluationService(search_fn, k=5).evaluate(_dataset())
        # AP/RR are per-query; aggregated renames them.
        assert "MAP" in report.aggregated
        assert "MRR" in report.aggregated
        assert "AP" not in report.aggregated
        assert "RR" not in report.aggregated

    def test_aggregated_means_are_arithmetic_mean(self):
        # Build a dataset where q1 hits at rank 1 and q2 misses.
        # MRR = (1.0 + 0.0) / 2 = 0.5
        search_fn = _make_search_fn({"q1": ["d1"], "q2": ["x"]})
        report = EvaluationService(search_fn, k=5).evaluate(_dataset())
        assert report.aggregated["MRR"] == pytest.approx(0.5)

    def test_empty_dataset_returns_empty_aggregated(self):
        search_fn = _make_search_fn({})
        empty_ds = EvaluationDataset(queries=[], qrels={})
        report = EvaluationService(search_fn).evaluate(empty_ds)
        assert report.aggregated == {}


class TestReportFormatting:
    def test_format_table_contains_metrics(self):
        search_fn = _make_search_fn({"q1": ["d1"], "q2": ["d4"]})
        report = EvaluationService(search_fn, k=5).evaluate(_dataset())
        text = report.format_table()
        for key in report.aggregated:
            assert key in text
        assert "queries: 2" in text
        assert "k=5" in text
