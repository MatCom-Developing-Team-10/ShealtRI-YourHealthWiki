"""Unit tests for QueryExpander and QueryExpansionPlugin.

No spaCy / heavy deps — uses hand-built ``IndexedCorpus`` objects so the
expansion logic is exercised in isolation.
"""

from __future__ import annotations

import pytest

from core.interfaces import IndexedCorpus, Plugin
from core.models import Document, PipelineContext, Query
from plugins.expansion import (
    ExpansionConfig,
    QueryExpander,
    QueryExpansionPlugin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _query_corpus(terms: list[str]) -> IndexedCorpus:
    """Build a tiny query corpus from a flat list of (lemmatised) tokens."""
    from collections import Counter

    inverted: dict[str, list[tuple[int, int]]] = {
        term: [(0, tf)] for term, tf in Counter(terms).items()
    }
    return IndexedCorpus(
        documents=[
            Document(
                doc_id="__query__",
                text=" ".join(terms),
                url="",
                metadata={"is_query": True},
            )
        ],
        processed_texts=[" ".join(terms)],
        inverted_index=inverted,
        vocabulary=sorted(inverted.keys()),
    )


class _FakeProcessor:
    """Deterministic stand-in for TextProcessor used in PRF tests.

    Lowercases, splits on whitespace and drops tokens shorter than 2 chars.
    Matches the behaviour the real TextProcessor would have on the toy
    inputs used here.
    """

    def process(self, text: str, is_query: bool = False) -> str:
        return " ".join(t for t in text.lower().split() if len(t) >= 2)


# ---------------------------------------------------------------------------
# Thesaurus-only expansion
# ---------------------------------------------------------------------------


class TestExpandWithThesaurus:
    def test_no_op_when_no_thesaurus_hits(self):
        expander = QueryExpander()
        original = _query_corpus(["zzz_no_term", "qqq_no_term"])
        out = expander.expand_with_thesaurus(original, target_vocabulary=None)
        # With no candidates, the expander returns the original corpus instance
        assert out is original

    def test_known_term_expands(self):
        expander = QueryExpander()
        original = _query_corpus(["hipertensión"])
        out = expander.expand_with_thesaurus(
            original,
            target_vocabulary=["hipertensión", "hta", "presión", "tensión", "arterial"],
        )
        added = set(out.vocabulary) - set(original.vocabulary)
        assert "hta" in added
        assert "presión" in added

    def test_target_vocabulary_filters_unknown_terms(self):
        expander = QueryExpander()
        original = _query_corpus(["hipertensión"])
        # Only 'hta' is in the document vocabulary; the rest must be dropped
        out = expander.expand_with_thesaurus(
            original, target_vocabulary=["hipertensión", "hta"]
        )
        added = set(out.vocabulary) - set(original.vocabulary)
        assert added == {"hta"}

    def test_no_target_vocabulary_keeps_everything(self):
        expander = QueryExpander()
        original = _query_corpus(["hipertensión"])
        out = expander.expand_with_thesaurus(original, target_vocabulary=None)
        added = set(out.vocabulary) - set(original.vocabulary)
        assert added  # at least one synonym registered for hipertensión

    def test_does_not_re_add_original_terms(self):
        expander = QueryExpander()
        original = _query_corpus(["hipertensión", "hta"])  # user wrote both
        out = expander.expand_with_thesaurus(original, target_vocabulary=None)
        # The originals must keep their TF; no duplicates in the new index.
        assert out.inverted_index["hipertensión"] == [(0, 1)]
        assert out.inverted_index["hta"] == [(0, 1)]

    def test_max_expanded_terms_caps_output(self):
        # 'diabetes' has many synonyms — cap at 2.
        expander = QueryExpander(ExpansionConfig(max_expanded_terms=2))
        original = _query_corpus(["diabetes"])
        out = expander.expand_with_thesaurus(original, target_vocabulary=None)
        added = set(out.vocabulary) - set(original.vocabulary)
        assert len(added) <= 2

    def test_use_thesaurus_disabled(self):
        expander = QueryExpander(
            ExpansionConfig(use_thesaurus=False, use_prf=False)
        )
        original = _query_corpus(["hipertensión"])
        out = expander.expand_with_thesaurus(original, target_vocabulary=None)
        assert out is original  # nothing to add

    def test_preserves_documents_and_processed_texts(self):
        expander = QueryExpander()
        original = _query_corpus(["hipertensión"])
        out = expander.expand_with_thesaurus(original, target_vocabulary=None)
        assert out.documents is original.documents
        assert out.processed_texts == original.processed_texts

    def test_does_not_mutate_input(self):
        expander = QueryExpander()
        original = _query_corpus(["hipertensión"])
        original_vocab_snapshot = list(original.vocabulary)
        original_index_snapshot = {
            k: list(v) for k, v in original.inverted_index.items()
        }
        expander.expand_with_thesaurus(original, target_vocabulary=None)
        assert original.vocabulary == original_vocab_snapshot
        assert original.inverted_index == original_index_snapshot


# ---------------------------------------------------------------------------
# Pseudo-Relevance Feedback
# ---------------------------------------------------------------------------


class TestExpandWithPRF:
    def _docs(self):
        return [
            Document(
                doc_id="d1",
                text="diabetes glucosa insulina páncreas diabetes glucosa",
                url="",
            ),
            Document(
                doc_id="d2",
                text="glucemia insulina hiperglucemia diabetes",
                url="",
            ),
        ]

    def test_prf_injects_top_terms_from_initial_results(self):
        expander = QueryExpander(
            ExpansionConfig(
                use_thesaurus=False, use_prf=True, prf_top_k=2, prf_terms_per_doc=2
            )
        )
        original = _query_corpus(["diabetes"])
        out = expander.expand_with_prf(
            original,
            initial_results=self._docs(),
            text_processor=_FakeProcessor(),
            target_vocabulary=None,
        )
        added = set(out.vocabulary) - set(original.vocabulary)
        # 'glucosa' and 'insulina' are the dominant terms across both docs
        assert "glucosa" in added or "insulina" in added

    def test_prf_with_no_initial_results_is_noop(self):
        expander = QueryExpander(
            ExpansionConfig(use_thesaurus=False, use_prf=True)
        )
        original = _query_corpus(["diabetes"])
        out = expander.expand_with_prf(
            original,
            initial_results=[],
            text_processor=_FakeProcessor(),
            target_vocabulary=None,
        )
        assert out is original

    def test_prf_respects_target_vocabulary(self):
        expander = QueryExpander(
            ExpansionConfig(use_thesaurus=False, use_prf=True)
        )
        original = _query_corpus(["diabetes"])
        out = expander.expand_with_prf(
            original,
            initial_results=self._docs(),
            text_processor=_FakeProcessor(),
            target_vocabulary=["diabetes", "glucosa"],  # only allow glucosa
        )
        added = set(out.vocabulary) - set(original.vocabulary)
        assert added <= {"glucosa"}

    def test_thesaurus_and_prf_combined(self):
        expander = QueryExpander(
            ExpansionConfig(use_thesaurus=True, use_prf=True, prf_top_k=2)
        )
        original = _query_corpus(["diabetes"])
        out = expander.expand_with_prf(
            original,
            initial_results=self._docs(),
            text_processor=_FakeProcessor(),
            target_vocabulary=None,
        )
        added = set(out.vocabulary) - set(original.vocabulary)
        # Thesaurus alone gives {dm, glucemia, glucosa, insulina, hiperglucemia}
        # PRF over the 2 docs reinforces glucosa/insulina.
        assert {"glucosa", "insulina"} <= added | original_set(original)


def original_set(corpus: IndexedCorpus) -> set[str]:
    return set(corpus.vocabulary)


# ---------------------------------------------------------------------------
# QueryExpansionPlugin
# ---------------------------------------------------------------------------


class TestQueryExpansionPlugin:
    def test_implements_plugin_contract(self):
        plugin = QueryExpansionPlugin()
        assert isinstance(plugin, Plugin)
        assert plugin.hook_name() == "pre_retrieval"

    def test_execute_mutates_query_indexed_corpus(self):
        plugin = QueryExpansionPlugin(target_vocabulary=["hipertensión", "hta"])
        query = Query(text="hipertensión", indexed_corpus=_query_corpus(["hipertensión"]))
        ctx = PipelineContext(query=query)

        out = plugin.execute(ctx)

        assert out is ctx
        assert ctx.query.indexed_corpus is not None
        assert "hta" in ctx.query.indexed_corpus.vocabulary

    def test_records_breadcrumb_in_metadata(self):
        plugin = QueryExpansionPlugin(target_vocabulary=["hipertensión", "hta", "presión"])
        query = Query(text="hipertensión", indexed_corpus=_query_corpus(["hipertensión"]))
        ctx = PipelineContext(query=query)

        out = plugin.execute(ctx)

        assert "expansion" in out.metadata
        info = out.metadata["expansion"]
        assert info["original_terms"] == ["hipertensión"]
        assert "hta" in info["added"]

    def test_skips_when_query_has_no_indexed_corpus(self):
        plugin = QueryExpansionPlugin()
        query = Query(text="x", indexed_corpus=None)
        ctx = PipelineContext(query=query)

        out = plugin.execute(ctx)

        assert out.query.indexed_corpus is None
        assert "expansion" not in out.metadata

    def test_no_target_vocabulary_still_works(self):
        plugin = QueryExpansionPlugin()  # no vocabulary filter
        query = Query(text="hipertensión", indexed_corpus=_query_corpus(["hipertensión"]))
        ctx = PipelineContext(query=query)

        out = plugin.execute(ctx)

        added = out.metadata["expansion"]["added"]
        assert added  # at least one synonym registered
