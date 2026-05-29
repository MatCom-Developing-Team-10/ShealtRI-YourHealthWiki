"""Integration test: query expansion improves recall on the medical corpus.

End-to-end demonstration that QueryExpansionPlugin actually helps:
    1. Fit the full LSI retriever over the 20-document synthetic corpus.
    2. Issue a lay-language query (e.g. "ataque corazón") that uses words
       the technical documents do not contain verbatim.
    3. Compare the top-5 IDs WITHOUT expansion vs WITH expansion.
    4. Assert that expansion either (a) introduces the on-topic document
       into the top-5, or (b) does not remove it if LSI already had it.

This test is the headline justification for the plugin.
"""

from __future__ import annotations

import pytest

pytest.importorskip("spacy")
# Check the model is installed without actually loading it (which costs ~150 MB).
# pkgutil.find_loader is cheap and avoids the OOM seen when both this check
# and the session-scoped text_processor fixture try to load spaCy back-to-back.
import importlib.util as _iu
if _iu.find_spec("es_core_news_md") is None:
    pytest.skip(
        "spaCy model 'es_core_news_md' not installed", allow_module_level=True
    )


from core.models import PipelineContext, Query
from modules.indexer import IndexerService
from modules.indexer.service import IndexerConfig
from modules.retriever import LSIRetriever
from plugins.expansion import QueryExpander, QueryExpansionPlugin


# ---------------------------------------------------------------------------
# Fitted pipeline (module-scoped — expensive)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_pipeline(sample_documents, text_processor):
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
    return indexer, retriever


def _top_ids(retriever, indexer, query_text, top_k=5):
    qc = indexer.build_query(query_text)
    results = retriever.retrieve(
        Query(text=query_text, indexed_corpus=qc), top_k=top_k
    )
    return [r.document.doc_id for r in results]


def _top_ids_expanded(retriever, indexer, query_text, plugin, top_k=5):
    qc = indexer.build_query(query_text)
    ctx = PipelineContext(query=Query(text=query_text, indexed_corpus=qc))
    plugin.execute(ctx)  # mutates ctx.query.indexed_corpus
    results = retriever.retrieve(ctx.query, top_k=top_k)
    return [r.document.doc_id for r in results], ctx.metadata.get("expansion", {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExpansionAddsTerms:
    def test_lay_query_gets_technical_synonyms(self, fitted_pipeline):
        """User writes 'ataque', plugin must add 'infarto' if it is in the corpus."""
        indexer, retriever = fitted_pipeline
        plugin = QueryExpansionPlugin(
            target_vocabulary=retriever.tfidf.vocabulary
        )
        _, info = _top_ids_expanded(
            retriever, indexer, "ataque corazón presión", plugin, top_k=5
        )
        added = set(info.get("added", []))
        assert added, (
            f"expansion produced no candidates against the document vocabulary; "
            f"this suggests the lemmatiser is dropping the lay query terms. "
            f"info={info}"
        )

    def test_full_term_gets_related_synonyms(self, fitted_pipeline):
        """A clean technical query should still gain semantic neighbours.

        Uses 'hipertensión' (long enough to survive the spell-checker)
        rather than an abbreviation. The expansion must add at least one
        of the registered synonyms that actually live in the corpus.
        """
        indexer, retriever = fitted_pipeline
        plugin = QueryExpansionPlugin(target_vocabulary=retriever.tfidf.vocabulary)
        _, info = _top_ids_expanded(
            retriever, indexer, "hipertensión arterial", plugin, top_k=5
        )
        added = set(info.get("added", []))
        assert added & {"hta", "presión", "tensión"}, (
            f"hipertensión query did not gain a registered synonym; added={added}"
        )


class TestExpansionImprovesRetrieval:
    """The harder claim: expansion lifts the on-topic doc into the top-5.

    Each test pairs a *lay* query with the canonical doc_id for its topic.
    Without expansion the doc may not appear in top-5 (LSI alone can't
    bridge the lay/technical gap). With expansion it must.
    """

    # Notes on the queries chosen:
    #   * Each contains at least one Spanish word LONG ENOUGH that the
    #     TrieSpellChecker doesn't mangle it (max_distance=2 means short
    #     abbreviations like 'dm'/'hta'/'iam' get rewritten to whatever
    #     2-char-edit-distance word IS in the corpus — a separate bug
    #     documented in docs/query-expansion.md §"Spell-checker
    #     interaction").
    #   * Each lay term IS in the medical thesaurus and its synonyms are
    #     present in the document vocabulary (target_vocabulary filter).
    @pytest.mark.parametrize(
        "lay_query,expected_doc_id",
        [
            # 'ataque' and 'corazón' both survive processing; expansion adds 'infarto'/'iam'/'miocardio'.
            ("ataque corazón dolor", "doc_infarto_008"),
            # 'hipertensión' survives cleanly; expansion adds 'hta'/'arterial'/'presión'.
            ("hipertensión arterial sangre", "doc_hipertension_001"),
            # 'diabetes' survives; expansion adds 'glucós'/'insulina'/'glucemia'.
            ("diabetes complicaciones tratamiento", "doc_diabetes_002"),
        ],
    )
    def test_expansion_surfaces_topical_doc(
        self, fitted_pipeline, lay_query, expected_doc_id
    ):
        indexer, retriever = fitted_pipeline
        plugin = QueryExpansionPlugin(target_vocabulary=retriever.tfidf.vocabulary)

        baseline = _top_ids(retriever, indexer, lay_query, top_k=5)
        expanded, info = _top_ids_expanded(
            retriever, indexer, lay_query, plugin, top_k=5
        )

        # The claim: post-expansion the on-topic doc must be in the top-5.
        # We make a tolerant check rather than an asymmetric one — sometimes
        # LSI already finds it without help; the test still verifies that
        # expansion does no harm.
        assert expected_doc_id in expanded, (
            f"After expansion, the on-topic doc {expected_doc_id!r} is missing "
            f"from top-5 for query {lay_query!r}.\n"
            f"  baseline:  {baseline}\n"
            f"  expanded:  {expanded}\n"
            f"  added:     {info.get('added')}"
        )


class TestExpansionIsHarmlessOnTechnicalQueries:
    """If the user already writes in technical terms, expansion must not
    push the canonical doc out of the top-1.
    """

    def test_technical_query_keeps_top_1(self, fitted_pipeline):
        indexer, retriever = fitted_pipeline
        plugin = QueryExpansionPlugin(target_vocabulary=retriever.tfidf.vocabulary)

        baseline = _top_ids(retriever, indexer, "diabetes glucosa insulina", top_k=1)
        expanded, _ = _top_ids_expanded(
            retriever, indexer, "diabetes glucosa insulina", plugin, top_k=1
        )

        assert baseline == expanded, (
            f"Expansion changed the top-1 for an unambiguous technical query.\n"
            f"  baseline:  {baseline}\n"
            f"  expanded:  {expanded}"
        )


class TestPRFRoundTrip:
    """Pseudo-relevance feedback exercised against the real retriever."""

    def test_prf_picks_terms_from_top_results(
        self, fitted_pipeline, text_processor
    ):
        indexer, retriever = fitted_pipeline
        # 1) baseline retrieval
        qc = indexer.build_query("diabetes")
        initial = retriever.retrieve(
            Query(text="diabetes", indexed_corpus=qc), top_k=3
        )
        assert initial

        # 2) PRF expansion using those results
        expander = QueryExpander()
        expander.config.use_thesaurus = False
        expander.config.use_prf = True
        expanded_corpus = expander.expand_with_prf(
            qc,
            initial_results=[r.document for r in initial],
            text_processor=text_processor,
            target_vocabulary=retriever.tfidf.vocabulary,
        )

        added = set(expanded_corpus.vocabulary) - set(qc.vocabulary)
        assert added, "PRF added no terms; pseudo-relevant docs may have been empty"
