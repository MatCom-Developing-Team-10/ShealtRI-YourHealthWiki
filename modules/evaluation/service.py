"""Evaluation service — runs the retriever over a test collection and reports metrics.

This is the orchestration layer. It is intentionally decoupled from any concrete
retriever: :class:`EvaluationService` only needs a ``search_fn`` that maps a query
string to a ranked list of original document ids. That keeps the metric logic
testable with a stub and lets the CLI wire in the real LSI retriever.

Run the bundled evaluation against the live corpus:

    python -m modules.evaluation.service
    python -m modules.evaluation.service --k 10

The CLI evaluates the **pure LSI retriever** (no web fallback) so the metrics
reflect the retrieval model itself and stay reproducible offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from .dataset import load_dataset
from .metrics import (
    average_precision,
    f1,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .models import EvaluationDataset, EvaluationReport, QueryEvalResult

# A search function maps (query_text, top_k) -> ranked list of original doc_ids.
SearchFn = Callable[[str, int], list[str]]


class EvaluationService:
    """Computes IR metrics for a retriever over an :class:`EvaluationDataset`.

    Args:
        search_fn: Callable returning a ranked list of original document ids for
            a given (query_text, top_k). Order matters — index 0 is rank 1.
        k: Cut-off rank for the @k metrics (P@k, R@k, NDCG@k).
    """

    def __init__(self, search_fn: SearchFn, k: int = 10) -> None:
        self._search_fn = search_fn
        self.k = k

    def evaluate(self, dataset: EvaluationDataset) -> EvaluationReport:
        """Run every query and aggregate the metrics.

        Args:
            dataset: Queries plus relevance judgments.

        Returns:
            An :class:`EvaluationReport` with per-query and aggregated metrics.
            Aggregated ``MAP`` is the mean of per-query Average Precision;
            aggregated ``MRR`` is the mean of per-query Reciprocal Rank.
        """
        per_query: list[QueryEvalResult] = []

        for query in dataset.queries:
            relevance = dataset.relevance_for(query.query_id)
            retrieved = self._search_fn(query.text, self.k)

            p_at_k = precision_at_k(retrieved, relevance, self.k)
            r_at_k = recall_at_k(retrieved, relevance, self.k)

            metrics = {
                f"P@{self.k}": p_at_k,
                f"R@{self.k}": r_at_k,
                f"F1@{self.k}": f1(p_at_k, r_at_k),
                f"NDCG@{self.k}": ndcg_at_k(retrieved, relevance, self.k),
                "AP": average_precision(retrieved, relevance),
                "RR": reciprocal_rank(retrieved, relevance),
            }

            num_relevant = sum(1 for g in relevance.values() if g >= 1)
            per_query.append(
                QueryEvalResult(
                    query_id=query.query_id,
                    text=query.text,
                    num_relevant=num_relevant,
                    num_retrieved=len(retrieved),
                    metrics=metrics,
                )
            )

        aggregated = self._aggregate(per_query)
        return EvaluationReport(per_query=per_query, aggregated=aggregated, k=self.k)

    def _aggregate(self, per_query: list[QueryEvalResult]) -> dict[str, float]:
        """Average each metric across queries; rename AP→MAP and RR→MRR."""
        if not per_query:
            return {}

        n = len(per_query)
        names = list(per_query[0].metrics.keys())
        means = {
            name: sum(r.metrics[name] for r in per_query) / n for name in names
        }

        # AP averaged over queries is MAP; RR averaged is MRR (lecture naming).
        means["MAP"] = means.pop("AP")
        means["MRR"] = means.pop("RR")
        return means


# ---------------------------------------------------------------------------
# CLI: wire the real LSI retriever and evaluate the bundled test collection
# ---------------------------------------------------------------------------

def _build_lsi_search_fn(pipeline, ranker=None) -> SearchFn:
    """Adapt the pipeline's pure LSI retriever to the SearchFn contract.

    Retrieval runs on chunks, so each returned chunk is mapped back to its
    original document id (stored in chunk metadata) and de-duplicated while
    preserving rank order — the evaluation judges documents, not chunks.

    Args:
        pipeline: The built pipeline exposing ``lsi`` and ``indexer``.
        ranker: Optional :class:`~core.interfaces.BaseRanker`. When provided,
            retrieved chunks are re-ranked before being mapped to documents, so
            the metrics reflect the LSI + ranker combination instead of LSI
            alone. This is what makes the ``--rerank`` comparison meaningful.
    """
    from core.models import Query

    def search(query_text: str, top_k: int) -> list[str]:
        # Retrieve more chunks than k, because several chunks can collapse to
        # the same source document after de-duplication.
        query_corpus = pipeline.indexer.build_query(query_text)
        query = Query(text=query_text, indexed_corpus=query_corpus)
        results = pipeline.lsi.retrieve(query, top_k=top_k * 5)

        if ranker is not None and results:
            results = ranker.rerank(query, results)

        ranked_docs: list[str] = []
        for r in results:
            original = r.document.metadata.get("original_doc_id")
            if not original:
                original = r.document.doc_id.split("__chunk_")[0]
            if original not in ranked_docs:
                ranked_docs.append(original)
        return ranked_docs

    return search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the LSI retriever against the bundled test collection.",
    )
    parser.add_argument(
        "--k", type=int, default=10, metavar="K",
        help="Cut-off rank for @k metrics (default: 10)",
    )
    parser.add_argument(
        "--queries", default="data/evaluation/eval_queries.json",
        help="Path to the queries JSON file.",
    )
    parser.add_argument(
        "--qrels", default="data/evaluation/eval_qrels.json",
        help="Path to the relevance judgments JSON file.",
    )
    parser.add_argument(
        "--per-query", action="store_true",
        help="Print per-query metrics in addition to the aggregate.",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Apply the HybridRanker (BM25+LSI) to retrieved docs before "
             "scoring, to compare against pure LSI.",
    )
    args = parser.parse_args()

    # Bootstrap sys.path so `python -m modules.evaluation.service` finds the root.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from cli import Pipeline  # reuse the existing pipeline wiring

    print("[eval] loading dataset...")
    dataset = load_dataset(args.queries, args.qrels)
    print(f"[eval] {len(dataset)} queries, "
          f"{sum(len(v) for v in dataset.qrels.values())} relevance judgments")

    print("[eval] building pipeline (indexing + LSI fit)...")
    pipeline = Pipeline()
    pipeline.build()

    ranker = None
    if args.rerank:
        from modules.ranker.service import HybridRanker
        ranker = HybridRanker()
        print("[eval] re-ranking enabled (HybridRanker BM25+LSI)")

    print(f"[eval] running evaluation (k={args.k})...\n")
    service = EvaluationService(_build_lsi_search_fn(pipeline, ranker=ranker), k=args.k)
    report = service.evaluate(dataset)

    if args.per_query:
        for result in report.per_query:
            print(f"  [{result.query_id}] {result.text[:50]!r} "
                  f"(rel={result.num_relevant}, ret={result.num_retrieved})")
            for name, value in result.metrics.items():
                print(f"      {name:<10} {value:.4f}")
            print()

    print(report.format_table())


if __name__ == "__main__":
    main()
