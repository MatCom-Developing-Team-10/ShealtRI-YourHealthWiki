"""Query expansion service for the SRI pipeline.

Two strategies, both opt-in via :class:`ExpansionConfig`:

1. **Thesaurus expansion** — deterministic lookup in
   :data:`plugins.expansion.thesaurus.MEDICAL_THESAURUS`. For each term in
   the query, register its registered synonyms / abbreviations / lay forms.

2. **Pseudo-Relevance Feedback (PRF)** — run the query, take the top-k
   results, extract their most frequent terms (excluding stopwords already
   removed by :class:`modules.text_processor.TextProcessor`), and inject
   them back as additional query terms. Useful when the user's wording is
   too vague to match the corpus directly.

Both strategies share the same downstream filter: an expanded term is
only kept if it exists in ``target_vocabulary`` (the document vocabulary
the TF-IDF processor was fitted on). Anything else would be silently
dropped by :class:`modules.retriever.tfidf_processor.TfidfProcessor.transform`
and is therefore pointless to add.

Expanded terms enter the query's inverted index with TF = 1 by default.
They are bounded by :attr:`ExpansionConfig.max_expanded_terms` to prevent
*query drift* (the failure mode where too much expansion changes the
intent of the original query).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol

from core.interfaces import IndexedCorpus, Plugin
from core.models import Document, PipelineContext

from .thesaurus import expand_term

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duck-typed protocol for the text processor (avoids a heavy import)
# ---------------------------------------------------------------------------


class _TextProcessorLike(Protocol):
    """The minimal interface QueryExpander needs from a TextProcessor.

    Declared as a Protocol so PRF stays decoupled from spaCy: any callable
    that lemmatises + tokenises a string into space-separated tokens works.
    """

    def process(self, text: str, is_query: bool = ...) -> str:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExpansionConfig:
    """Knobs for the two query-expansion strategies.

    Attributes:
        use_thesaurus: Enable thesaurus lookup for each original term.
        use_prf: Enable pseudo-relevance feedback. Requires a retriever and
            a text processor to be passed to :meth:`QueryExpander.expand`.
        prf_top_k: How many initial results to use as pseudo-relevant.
        prf_terms_per_doc: How many top terms to extract from each
            pseudo-relevant document.
        max_expanded_terms: Hard cap on the total number of terms added to
            the original query. The most frequent / first-seen are kept.
        expanded_term_tf: TF value assigned to expanded terms in the new
            inverted index. Set < 1 to under-weight expansions vs the user's
            actual words; the implementation stores integers, so values below
            1 are clamped to 1 and tuning is done via ``max_expanded_terms``.
    """

    use_thesaurus: bool = True
    use_prf: bool = False
    prf_top_k: int = 3
    prf_terms_per_doc: int = 3
    max_expanded_terms: int = 6
    expanded_term_tf: int = 1


# ---------------------------------------------------------------------------
# Core service
# ---------------------------------------------------------------------------


class QueryExpander:
    """Produces an enriched :class:`IndexedCorpus` from a query corpus.

    The expander never mutates its inputs; ``expand_*`` methods always
    return a new :class:`IndexedCorpus`. Documents and processed_texts on
    the query corpus are passed through unchanged — only the
    ``inverted_index`` and ``vocabulary`` are extended.

    Example:
        config = ExpansionConfig(use_thesaurus=True)
        expander = QueryExpander(config)

        # Indexer produces the query corpus from the raw user query
        query_corpus = indexer.build_query("ataque al corazón")
        # tfidf.vocabulary is the document vocabulary; expansions outside
        # this set would be filtered by TfidfProcessor.transform() anyway.
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
        """Expand using only the static medical thesaurus.

        Args:
            query_corpus: Output of ``IndexerService.build_query()``.
            target_vocabulary: Document vocabulary. Expanded terms not in
                this set are dropped. ``None`` disables filtering.

        Returns:
            New IndexedCorpus with extra terms in the inverted index.
        """
        return self._build_expanded(
            query_corpus=query_corpus,
            extra_terms=set(),
            target_vocabulary=target_vocabulary,
        )

    def expand_with_prf(
        self,
        query_corpus: IndexedCorpus,
        initial_results: list[Document],
        text_processor: _TextProcessorLike,
        target_vocabulary: Iterable[str] | None = None,
    ) -> IndexedCorpus:
        """Expand using pseudo-relevance feedback over an initial result set.

        Args:
            query_corpus: Output of ``IndexerService.build_query()``.
            initial_results: Top-k documents returned by the retriever for
                the *original* query.
            text_processor: Same TextProcessor instance used during
                indexing (so PRF extracts the same lemmas the corpus uses).
            target_vocabulary: Document vocabulary filter (see above).

        Returns:
            New IndexedCorpus with original + PRF terms (and thesaurus
            terms if ``config.use_thesaurus`` is also True).
        """
        prf_terms = self._extract_prf_terms(initial_results, text_processor)
        return self._build_expanded(
            query_corpus=query_corpus,
            extra_terms=prf_terms,
            target_vocabulary=target_vocabulary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_prf_terms(
        self,
        documents: list[Document],
        text_processor: _TextProcessorLike,
    ) -> set[str]:
        """Pick the most frequent terms across the top-k pseudo-relevant docs.

        Each document contributes its ``prf_terms_per_doc`` most frequent
        terms; the global Counter then picks the most-shared terms across
        the document set. This guards against a single very long document
        dominating the expansion.
        """
        if not documents or self.config.prf_top_k <= 0:
            return set()

        ballot: Counter[str] = Counter()
        for doc in documents[: self.config.prf_top_k]:
            processed = text_processor.process(doc.text, is_query=False)
            if not processed:
                continue
            local_counts = Counter(processed.split())
            for term, _ in local_counts.most_common(self.config.prf_terms_per_doc):
                ballot[term] += 1

        return {term for term, _ in ballot.most_common(self.config.max_expanded_terms)}

    def _build_expanded(
        self,
        query_corpus: IndexedCorpus,
        extra_terms: set[str],
        target_vocabulary: Iterable[str] | None,
    ) -> IndexedCorpus:
        original_terms = set(query_corpus.vocabulary)

        # Collect candidate expansions
        candidates: set[str] = set(extra_terms)
        if self.config.use_thesaurus:
            for term in original_terms:
                candidates.update(expand_term(term))

        # Never re-add a term the user already used
        candidates -= original_terms

        # Filter against the document vocabulary if provided
        if target_vocabulary is not None:
            allowed = set(target_vocabulary)
            candidates &= allowed

        # Cap to avoid query drift
        if len(candidates) > self.config.max_expanded_terms:
            # Deterministic truncation: sorted alphabetically
            candidates = set(sorted(candidates)[: self.config.max_expanded_terms])

        if not candidates:
            logger.debug(
                "QueryExpander: no expansion candidates for %s", sorted(original_terms)
            )
            return query_corpus  # No-op: return original unchanged

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

    Only the thesaurus strategy is wired in by default — PRF needs a
    retriever, which the plugin does not own. Callers that want PRF should
    instantiate :class:`QueryExpander` directly and call
    :meth:`QueryExpander.expand_with_prf`.
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
