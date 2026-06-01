"""Data models for the evaluation module.

These containers carry the evaluation dataset (queries + relevance judgments)
and the computed results, keeping :mod:`modules.evaluation.metrics` (pure math)
decoupled from :mod:`modules.evaluation.service` (orchestration).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalQuery:
    """A single evaluation query.

    Attributes:
        query_id: Stable identifier used to join the query with its qrels.
        text: Natural-language query string fed to the pipeline.
    """

    query_id: str
    text: str


@dataclass(slots=True)
class EvaluationDataset:
    """A test collection: queries plus their relevance judgments (qrels).

    Attributes:
        queries: List of evaluation queries.
        qrels: Relevance judgments as ``{query_id: {doc_id: grade}}``. A grade
            >= 1 marks a relevant document; the magnitude (e.g. 1 vs 2) encodes
            graded relevance used by NDCG. A missing doc_id means grade 0.
    """

    queries: list[EvalQuery]
    qrels: dict[str, dict[str, int]]

    def relevance_for(self, query_id: str) -> dict[str, int]:
        """Return the relevance map for a query, or an empty map if absent."""
        return self.qrels.get(query_id, {})

    def __len__(self) -> int:
        return len(self.queries)


@dataclass(slots=True)
class QueryEvalResult:
    """Per-query metric values produced by the evaluation run.

    Attributes:
        query_id: Identifier of the evaluated query.
        text: The query text (kept for human-readable reports).
        num_relevant: Number of documents judged relevant for this query.
        num_retrieved: Number of documents the system returned.
        metrics: Mapping of metric name -> value (all in [0, 1] except DCG).
    """

    query_id: str
    text: str
    num_relevant: int
    num_retrieved: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationReport:
    """Aggregated evaluation results across all queries.

    Attributes:
        per_query: Individual results, one per evaluated query.
        aggregated: Mean of each metric across queries. MAP and MRR are the
            aggregated forms of average_precision and reciprocal_rank.
        k: The cut-off rank used for the @k metrics (for display).
    """

    per_query: list[QueryEvalResult]
    aggregated: dict[str, float]
    k: int

    def format_table(self) -> str:
        """Render the aggregated metrics as a readable text block."""
        lines = [
            "=" * 52,
            f"  EVALUATION REPORT  (queries: {len(self.per_query)}, k={self.k})",
            "=" * 52,
        ]
        for name, value in self.aggregated.items():
            lines.append(f"  {name:<22} {value:.4f}")
        lines.append("=" * 52)
        return "\n".join(lines)
