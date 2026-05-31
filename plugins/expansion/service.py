"""Query expansion service for the SRI pipeline.

Implements a single deterministic strategy:

**Thesaurus expansion** — for each term in the user's query, look it up in
:data:`plugins.expansion.thesaurus.MEDICAL_THESAURUS` and inject its
registered synonyms / abbreviations / lay forms into the query's
inverted index.

An expanded term is only kept if it exists in ``target_vocabulary`` (the
document vocabulary the TF-IDF processor was fitted on). Anything else
would be silently dropped by
:class:`modules.retriever.tfidf_processor.TfidfProcessor.transform`
and is therefore pointless to add.

Expanded terms enter the query's inverted index with TF = 1 by default.
They are bounded by :attr:`ExpansionConfig.max_expanded_terms` to prevent
*query drift* (the failure mode where too much expansion changes the
intent of the original query).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from core.interfaces import IndexedCorpus, Plugin
from core.models import PipelineContext

from .thesaurus import expand_term

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExpansionConfig:
    """Knobs for the thesaurus expansion strategy.

    Attributes:
        use_thesaurus: Enable thesaurus lookup for each original term.
        max_expanded_terms: Hard cap on the total number of terms added to
            the original query. Sorted alphabetically when truncating, so
            the cap is deterministic.
        expanded_term_tf: TF value assigned to expanded terms in the new
            inverted index. Values below 1 are clamped to 1; tuning is done
            via ``max_expanded_terms``.
    """

    use_thesaurus: bool = True
    max_expanded_terms: int = 6
    expanded_term_tf: int = 1


# ---------------------------------------------------------------------------
# Core service
# ---------------------------------------------------------------------------


class QueryExpander:
    """Produces an enriched :class:`IndexedCorpus` from a query corpus.

    The expander never mutates its inputs; ``expand_with_thesaurus``
    returns a new :class:`IndexedCorpus`. Documents and processed_texts
    on the query corpus are passed through unchanged — only the
    ``inverted_index`` and ``vocabulary`` are extended.

    Example:
        config = ExpansionConfig(use_thesaurus=True)
        expander = QueryExpander(config)

        query_corpus = indexer.build_query("ataque al corazón")
        expanded = expander.expand_with_thesaurus(
            query_corpus, target_vocabulary=tfidf.vocabulary
        )
        query.indexed_corpus = expanded
        results = retriever.retrieve(query)
    """

    def __init__(self, config: ExpansionConfig | None = None) -> None:
        self.config = config or ExpansionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand_with_thesaurus(
        self,
        query_corpus: IndexedCorpus,
        target_vocabulary: Iterable[str] | None = None,
    ) -> IndexedCorpus:
        """Expand using the static medical thesaurus.

        Args:
            query_corpus: Output of ``IndexerService.build_query()``.
            target_vocabulary: Document vocabulary. Expanded terms not in
                this set are dropped. ``None`` disables filtering.

        Returns:
            New IndexedCorpus with extra terms in the inverted index, or
            the original corpus instance when no candidate survives.
        """
        original_terms = set(query_corpus.vocabulary)

        candidates: set[str] = set()
        if self.config.use_thesaurus:
            for term in original_terms:
                candidates.update(expand_term(term))

        # Never re-add a term the user already used
        candidates -= original_terms

        # Filter against the document vocabulary if provided
        if target_vocabulary is not None:
            allowed = set(target_vocabulary)
            candidates &= allowed

        # Cap to avoid query drift (deterministic alphabetical truncation)
        if len(candidates) > self.config.max_expanded_terms:
            candidates = set(sorted(candidates)[: self.config.max_expanded_terms])

        if not candidates:
            logger.debug(
                "QueryExpander: no expansion candidates for %s", sorted(original_terms)
            )
            return query_corpus

        # Build the new inverted index. Postings of original terms are
        # copied verbatim; expanded terms get a single posting with TF.
        tf = max(1, self.config.expanded_term_tf)
        new_inverted: dict[str, list[tuple[int, int]]] = {
            term: list(postings)
            for term, postings in query_corpus.inverted_index.items()
        }
        for term in candidates:
            new_inverted[term] = [(0, tf)]

        new_vocab = sorted(new_inverted.keys())

        logger.info(
            "QueryExpander: expanded %d original term(s) with %d new term(s): %s",
            len(original_terms), len(candidates), sorted(candidates),
        )

        return IndexedCorpus(
            documents=query_corpus.documents,
            processed_texts=query_corpus.processed_texts,
            inverted_index=new_inverted,
            vocabulary=new_vocab,
        )


# ---------------------------------------------------------------------------
# Plugin wrapper
# ---------------------------------------------------------------------------


class QueryExpansionPlugin(Plugin):
    """Plug-and-play wrapper that registers :class:`QueryExpander` on the
    ``pre_retrieval`` hook.

    The plugin reads ``context.query.indexed_corpus`` (built by the indexer
    earlier in the pipeline) and replaces it with the expanded version.
    The original term list is preserved under ``context.metadata['expansion']``
    so downstream code (or the UI) can show "we also searched for X, Y, Z".
    """

    def __init__(
        self,
        expander: QueryExpander | None = None,
        target_vocabulary: Iterable[str] | None = None,
    ) -> None:
        """Initialize the plugin.

        Args:
            expander: Custom :class:`QueryExpander`. Defaults to one with
                ``use_thesaurus=True``.
            target_vocabulary: Document vocabulary used to filter expanded
                terms. Typically passed as ``tfidf.vocabulary`` after the
                retriever has been fitted.
        """
        self.expander = expander or QueryExpander()
        self.target_vocabulary = (
            list(target_vocabulary) if target_vocabulary is not None else None
        )

    def hook_name(self) -> str:
        return "pre_retrieval"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.query.indexed_corpus is None:
            logger.warning(
                "QueryExpansionPlugin: query has no indexed_corpus; skipping"
            )
            return context

        original = context.query.indexed_corpus
        expanded = self.expander.expand_with_thesaurus(
            original, target_vocabulary=self.target_vocabulary
        )

        added = sorted(set(expanded.vocabulary) - set(original.vocabulary))
        context.query.indexed_corpus = expanded
        context.metadata["expansion"] = {
            "original_terms": sorted(original.vocabulary),
            "expanded_terms": sorted(expanded.vocabulary),
            "added": added,
        }
        return context
