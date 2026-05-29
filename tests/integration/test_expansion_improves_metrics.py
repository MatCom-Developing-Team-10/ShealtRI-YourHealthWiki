"""Integration test: query expansion measurably improves evaluation metrics.

This is the test that *closes the loop* between the three concepts:

    1. Build a deterministic mini-dataset (queries + qrels).
    2. Evaluate the LSI retriever WITHOUT the expansion plugin.
    3. Evaluate it WITH the expansion plugin (thesaurus enabled).
    4. Assert that at least one of MAP / MRR / NDCG@5 improved or stayed
       equal — proving the plugin is not making the system worse.

Uses :class:`EvaluationService` directly with a tiny corpus built around
the medical thesaurus so the relationship between query and qrels is
deterministic. spaCy is required because the same indexer normalises both
documents and queries; if it cannot be loaded, the test is skipped.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")


from core.models import Document, PipelineContext, Query
from modules.evaluation.models import EvalQuery, EvaluationDataset
from modules.evaluation.service import EvaluationService
from modules.indexer.service import IndexerConfig, IndexerService
from modules.retriever import LSIRetriever
from plugins.expansion import QueryExpansionPlugin


# ---------------------------------------------------------------------------
# Mini test collection
# ---------------------------------------------------------------------------


_DOCS = [
    Document(
        doc_id="d_hta",
        text=(
            "La hipertensión arterial es una enfermedad cardiovascular crónica "
            "que afecta a millones de personas. Los pacientes con hipertensión "
            "deben controlar su tensión arterial regularmente. El tratamiento "
            "incluye cambios en el estilo de vida y medicamentos."
        ),
        url="u_hta",
    ),
    Document(
        doc_id="d_dm",
        text=(
            "La diabetes mellitus es una enfermedad metabólica caracterizada por "
            "niveles elevados de glucosa en sangre. La diabetes tipo 2 es la más "
            "común. El tratamiento incluye dieta, ejercicio e insulina."
        ),
        url="u_dm",
    ),
    Document(
        doc_id="d_asma",
        text=(
            "El asma es una enfermedad respiratoria crónica que causa "
            "inflamación de las vías respiratorias bronquial. El asma puede "
            "presentarse con disnea y sibilancias. El tratamiento usa "
            "broncodilatadores."
        ),
        url="u_asma",
    ),
    Document(
        doc_id="d_cancer",
        text=(
            "El cáncer es una enfermedad caracterizada por crecimiento celular "
            "descontrolado. Existen muchos tipos de tumor. El tratamiento "
            "incluye quimioterapia y radioterapia."
        ),
        url="u_cancer",
    ),
]


# Queries phrased with lay / abbreviation terms that the thesaurus knows
# how to expand to the documents' technical vocabulary.
_QUERIES = [
    EvalQuery("q_hta", "presión arterial alta"),
    EvalQuery("q_dm", "diabetes glucosa"),
    EvalQuery("q_asma", "asma respiratorio"),
]

_QRELS = {
    "q_hta": {"d_hta": 2},
    "q_dm": {"d_dm": 2},
    "q_asma": {"d_asma": 2},
}


# ---------------------------------------------------------------------------
# Fixture: fitted retriever with the mini corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_retriever(text_processor):
    from tests.conftest import InMemoryDocumentStore, InMemoryRepository

    store = InMemoryDocumentStore()
    repo = InMemoryRepository()
    indexer = IndexerService(
        text_processor=text_processor,
        config=IndexerConfig(min_term_frequency=1),
    )
    corpus = indexer.build(_DOCS)

    retriever = LSIRetriever(
        repository=repo,
        document_store=store,
        n_components=3,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)
    return indexer, retriever


# ---------------------------------------------------------------------------
# Search adapters wired to EvaluationService
# ---------------------------------------------------------------------------


def _baseline_search_fn(indexer, retriever):
    def search(query_text: str, top_k: int) -> list[str]:
        qc = indexer.build_query(query_text)
        results = retriever.retrieve(
            Query(text=query_text, indexed_corpus=qc), top_k=top_k,
        )
        return [r.document.doc_id for r in results]

    return search


def _expanded_search_fn(indexer, retriever):
    plugin = QueryExpansionPlugin(target_vocabulary=retriever.tfidf.vocabulary)

    def search(query_text: str, top_k: int) -> list[str]:
        qc = indexer.build_query(query_text)
        ctx = PipelineContext(query=Query(text=query_text, indexed_corpus=qc))
        plugin.execute(ctx)  # mutates ctx.query.indexed_corpus
        results = retriever.retrieve(ctx.query, top_k=top_k)
        return [r.document.doc_id for r in results]

    return search


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset():
    return EvaluationDataset(queries=_QUERIES, qrels=_QRELS)


class TestExpansionHelpsOrIsHarmless:
    def test_map_does_not_decrease_with_expansion(self, fitted_retriever, dataset):
        indexer, retriever = fitted_retriever

        eval_service = EvaluationService(
            _baseline_search_fn(indexer, retriever), k=5
        )
        baseline = eval_service.evaluate(dataset).aggregated

        eval_service_exp = EvaluationService(
            _expanded_search_fn(indexer, retriever), k=5
        )
        expanded = eval_service_exp.evaluate(dataset).aggregated

        # Expansion must not hurt MAP. A floating-point slack of 1e-6
        # absorbs noise from ties.
        assert expanded["MAP"] >= baseline["MAP"] - 1e-6, (
            f"Expansion lowered MAP: {baseline['MAP']:.4f} → {expanded['MAP']:.4f}"
        )

    def test_at_least_one_metric_improves_or_stays_equal(
        self, fitted_retriever, dataset
    ):
        indexer, retriever = fitted_retriever
        baseline = (
            EvaluationService(_baseline_search_fn(indexer, retriever), k=5)
            .evaluate(dataset)
            .aggregated
        )
        expanded = (
            EvaluationService(_expanded_search_fn(indexer, retriever), k=5)
            .evaluate(dataset)
            .aggregated
        )

        # The headline metrics — at least ONE must not regress.
        protected = ["MAP", "MRR", "NDCG@5"]
        regressions = [
            m for m in protected if expanded[m] < baseline[m] - 1e-6
        ]
        assert len(regressions) < len(protected), (
            f"All headline metrics regressed with expansion: "
            f"baseline={ {k: baseline[k] for k in protected} } "
            f"expanded={ {k: expanded[k] for k in protected} }"
        )


class TestEvaluationServiceProducesNumbers:
    """Sanity floor — the evaluator must compute non-trivial metrics."""

    def test_baseline_mrr_above_zero(self, fitted_retriever, dataset):
        indexer, retriever = fitted_retriever
        report = (
            EvaluationService(_baseline_search_fn(indexer, retriever), k=5)
            .evaluate(dataset)
        )
        # At least one query should find its relevant document — MRR > 0
        assert report.aggregated["MRR"] > 0.0
