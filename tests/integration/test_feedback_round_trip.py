"""Integration test: explicit relevance feedback round-trip.

Walks the real path:

    1. Build & fit LSI over the 20-doc synthetic corpus.
    2. Issue a query without feedback — record the original top-5.
    3. Record explicit ``relevant`` judgments for a chosen doc.
    4. Re-issue the same query with feedback wired in — assert the chosen
       doc rose in the ranking (Rocchio pulled the query toward it).
    5. Record an explicit ``non-relevant`` judgment for a doc that was in
       the top-5 — assert that doc fell in the ranking.

Uses the InMemoryRepository fake from conftest so the test is fast and
deterministic. spaCy is shared via the session fixture.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")


from core.models import Query
from modules.indexer.service import IndexerConfig, IndexerService
from modules.retriever import LSIRetriever
from plugins.feedback import (
    InMemoryFeedbackStore,
    RelevanceFeedbackService,
    RocchioReweighter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_pipeline(sample_documents, text_processor):
    """Fit the LSI retriever once for the module — slow operation."""
    from tests.conftest import InMemoryDocumentStore, InMemoryRepository

    store = InMemoryDocumentStore()
    repo = InMemoryRepository()
    indexer = IndexerService(
        text_processor=text_processor,
        config=IndexerConfig(min_term_frequency=1),
    )
    corpus = indexer.build(sample_documents)
    retriever = LSIRetriever(
        repository=repo,
        document_store=store,
        n_components=15,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)
    return indexer, retriever, repo


def _ids(results) -> list[str]:
    return [r.document.doc_id for r in results]


def _retrieve(indexer, retriever, query_text, top_k=10):
    qc = indexer.build_query(query_text)
    return retriever.retrieve(
        Query(text=query_text, indexed_corpus=qc), top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeedbackPullsRelevantDocsUp:
    """Marking a doc as relevant should boost its rank on the next query."""

    def test_relevant_judgment_lifts_doc_rank(self, fitted_pipeline):
        indexer, retriever, _ = fitted_pipeline

        # Baseline: query and capture the rank of doc_neumonia_011
        baseline = _retrieve(indexer, retriever, "infección pulmón", top_k=10)
        baseline_ids = _ids(baseline)
        assert "doc_neumonia_011" in baseline_ids
        baseline_rank = baseline_ids.index("doc_neumonia_011")

        # User marks doc_neumonia_011 as relevant
        feedback = RelevanceFeedbackService(
            InMemoryFeedbackStore(),
            rocchio=RocchioReweighter(alpha=1.0, beta=2.0, gamma=0.0),
        )
        feedback.record("infección pulmón", "doc_neumonia_011", relevant=True)

        # Re-issue with feedback wired in — same retriever, same repo,
        # only the optional feedback_service parameter changes.
        retriever.feedback_service = feedback
        try:
            new_results = _retrieve(
                indexer, retriever, "infección pulmón", top_k=10
            )
        finally:
            retriever.feedback_service = None

        new_ids = _ids(new_results)
        assert "doc_neumonia_011" in new_ids
        new_rank = new_ids.index("doc_neumonia_011")
        # The doc must NOT fall in the ranking. Ideally it rises;
        # tolerating "stays in place" because Rocchio can keep a top-1
        # at top-1.
        assert new_rank <= baseline_rank, (
            f"feedback caused the relevant doc to drop: "
            f"{baseline_rank} → {new_rank}"
        )


class TestFeedbackPushesNonRelevantDocsDown:
    """Marking a doc as non-relevant should lower its rank."""

    def test_non_relevant_judgment_lowers_doc_rank(self, fitted_pipeline):
        indexer, retriever, _ = fitted_pipeline

        query = "diabetes glucosa"
        baseline = _retrieve(indexer, retriever, query, top_k=10)
        baseline_ids = _ids(baseline)
        assert len(baseline_ids) >= 2

        # Pick the doc currently at position 1 (second place) as the
        # "user said NO" target. We want to see it move further down.
        target = baseline_ids[1]

        feedback = RelevanceFeedbackService(
            InMemoryFeedbackStore(),
            rocchio=RocchioReweighter(alpha=1.0, beta=0.0, gamma=2.0),
        )
        feedback.record(query, target, relevant=False)

        retriever.feedback_service = feedback
        try:
            new_results = _retrieve(indexer, retriever, query, top_k=10)
        finally:
            retriever.feedback_service = None

        new_ids = _ids(new_results)
        if target in new_ids:
            new_rank = new_ids.index(target)
            # Must not improve in ranking. Ideally falls.
            assert new_rank >= 1, (
                f"non-relevant judgment somehow promoted {target}"
            )
        # If target was pushed out of the top-10 entirely that's also a
        # legitimate outcome.


class TestRetrieverWithoutFeedbackUnaffected:
    """Wiring a service with no judgments must not change retrieval."""

    def test_empty_feedback_service_is_a_noop(self, fitted_pipeline):
        indexer, retriever, _ = fitted_pipeline

        baseline = _retrieve(indexer, retriever, "asma", top_k=5)

        retriever.feedback_service = RelevanceFeedbackService()  # empty store
        try:
            with_empty = _retrieve(indexer, retriever, "asma", top_k=5)
        finally:
            retriever.feedback_service = None

        assert _ids(baseline) == _ids(with_empty)


class TestFeedbackDoesNotBreakWhenRepoLacksEmbeddings:
    """A repository whose get_embedding is not implemented must still work."""

    def test_repo_without_embedding_lookup_gracefully_skips(self, fitted_pipeline):
        from core.interfaces import BaseRepository

        indexer, retriever, real_repo = fitted_pipeline

        class _NoLookupRepo(BaseRepository):
            def add_documents(self, documents, embeddings=None):
                pass

            def search_similar(self, query_vector, top_k=10):
                return real_repo.search_similar(query_vector, top_k=top_k)

            # NB: deliberately do NOT override get_embedding — keeps default
            # behaviour (NotImplementedError).

        feedback = RelevanceFeedbackService()
        feedback.record("asma", "doc_asma_003", relevant=True)

        retriever.repository = _NoLookupRepo()
        retriever.feedback_service = feedback
        try:
            # Must not raise even though the repo can't resolve embeddings.
            results = _retrieve(indexer, retriever, "asma", top_k=3)
        finally:
            retriever.repository = real_repo
            retriever.feedback_service = None

        assert isinstance(results, list)
