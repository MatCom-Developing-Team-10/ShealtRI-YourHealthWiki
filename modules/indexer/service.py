"""Indexer service — builds IndexedCorpus from preprocessed documents.

This module bridges document loading and the LSI retriever:
    Documents → TextProcessor → IndexerService.build() → IndexedCorpus → TfidfProcessor

Responsibilities:
    1. Orchestrate lexical preprocessing via TextProcessor.
    2. Build the inverted index: term → [(doc_index, term_frequency), ...].
    3. Manage the vocabulary (sorted, deterministic column order for TF-IDF).
    4. Apply collection-level filters (minimum document length, minimum term
       frequency) before the matrix is built.
    5. Provide a dual path:
        - build(documents)  → indexation path; tokens fed to the spell checker.
        - build_query(text) → query path; tokens corrected against the spell
          checker vocabulary.

Flow contracts:
    - TextProcessor.process(text, is_query=False) must be called for documents so
      that their tokens populate the Trie used to correct later queries.
    - TextProcessor.process(text, is_query=True) must be called for queries so
      that OOV tokens are rewritten to the closest known vocabulary term.
    - The returned IndexedCorpus satisfies its `documents`/`processed_texts`
      length invariant (see core.interfaces.IndexedCorpus).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

from core.interfaces import IndexedCorpus
from core.models import Document
from modules.text_processor import TextProcessor


logger = logging.getLogger(__name__)

# Synthetic doc_id used when wrapping a query inside an IndexedCorpus.
# The TF-IDF transform path only reads inverted_index, but IndexedCorpus
# still requires a documents list of matching length to satisfy its invariant.
_QUERY_DOC_ID = "__query__"


@dataclass
class IndexerConfig:
    """Configuration for the indexer service.

    Attributes:
        min_document_length: Minimum number of processed tokens required to
            keep a document in the corpus. Documents below this threshold are
            silently dropped (and skipped in the inverted index).
        min_term_frequency: Minimum total corpus frequency for a term to be
            kept in the vocabulary. Filters rare typos/noise before TF-IDF.
        log_progress_every: Emit an INFO log every N processed documents.
    """

    min_document_length: int = 1
    min_term_frequency: int = 1
    log_progress_every: int = 100


class IndexerService:
    """Builds IndexedCorpus objects from raw documents and from raw query text.

    The service is stateless across invocations: every call to build() or
    build_query() produces a fresh IndexedCorpus. Persistent state lives in
    the injected TextProcessor (it owns the spell checker vocabulary that
    accumulates across build() calls and is consulted by build_query()).

    Example:
        processor = TextProcessor()
        indexer = IndexerService(text_processor=processor)

        corpus = indexer.build(documents)          # fills spell vocab
        query_corpus = indexer.build_query("hipertensoin arterail")
        # query tokens corrected against the accumulated vocab
    """

    def __init__(
        self,
        text_processor: TextProcessor,
        config: IndexerConfig | None = None,
    ) -> None:
        """Initialize the indexer service.

        Args:
            text_processor: TextProcessor instance for preprocessing. The same
                instance must be reused for build() and build_query() so that
                the spell checker vocabulary is shared.
            config: Indexer configuration. Uses defaults if None.
        """
        self.text_processor = text_processor
        self.config = config or IndexerConfig()

    # ------------------------------------------------------------------
    # Indexation path
    # ------------------------------------------------------------------

    def build(self, documents: list[Document]) -> IndexedCorpus:
        """Build an IndexedCorpus from a collection of documents.

        Pipeline:
            1. Preprocess each document (is_query=False → tokens added to
               the spell checker vocabulary by TextProcessor).
            2. Drop documents with fewer than ``min_document_length`` tokens.
            3. Compute per-document term frequencies.
            4. Aggregate into the inverted index.
            5. Drop terms whose total corpus frequency is below
               ``min_term_frequency``.
            6. Produce a sorted vocabulary to guarantee a deterministic
               column order in the downstream TF-IDF matrix.

        Args:
            documents: Documents to index. May be empty.

        Returns:
            IndexedCorpus with aligned documents/processed_texts, the inverted
            index, and the sorted vocabulary. Empty input yields an empty
            (but valid) IndexedCorpus.
        """
        if not documents:
            logger.warning("build() called with an empty document list")
            return IndexedCorpus(
                documents=[],
                processed_texts=[],
                inverted_index={},
                vocabulary=[],
            )

        kept_documents: list[Document] = []
        kept_processed_texts: list[str] = []
        per_doc_tf: list[Counter[str]] = []

        total = len(documents)
        for position, doc in enumerate(documents, start=1):
            processed = self.text_processor.process(doc.text, is_query=False)
            tokens = processed.split() if processed else []

            if len(tokens) < self.config.min_document_length:
                logger.debug(
                    "Skipping document '%s': %d tokens < min_document_length=%d",
                    doc.doc_id, len(tokens), self.config.min_document_length,
                )
                continue

            kept_documents.append(doc)
            kept_processed_texts.append(processed)
            per_doc_tf.append(Counter(tokens))

            if position % self.config.log_progress_every == 0:
                logger.info("Indexed %d/%d documents", position, total)

        inverted_index, vocabulary = self._aggregate_postings(per_doc_tf)

        logger.info(
            "Indexing complete: %d/%d documents kept, vocabulary size=%d",
            len(kept_documents), total, len(vocabulary),
        )

        return IndexedCorpus(
            documents=kept_documents,
            processed_texts=kept_processed_texts,
            inverted_index=inverted_index,
            vocabulary=vocabulary,
        )

    # ------------------------------------------------------------------
    # Incremental update path
    # ------------------------------------------------------------------

    def update(
        self,
        existing: IndexedCorpus,
        new_documents: list[Document],
    ) -> IndexedCorpus:
        """Extend an existing IndexedCorpus with new documents.

        Used when the crawler has produced additional documents and we want
        to index them without re-processing the entire corpus. The existing
        inverted index is preserved; new postings are appended with doc
        indices assigned continuously after the current range.

        Idempotent by doc_id: documents whose doc_id is already in
        ``existing`` are silently skipped, never duplicated.

        Note:
            ``min_term_frequency`` is **not** re-applied here — doing so
            would retroactively drop terms that were valid at build time.
            Call ``build()`` from scratch if you need to re-apply filters.

        Args:
            existing: Current IndexedCorpus snapshot.
            new_documents: Candidate documents to add.

        Returns:
            A new IndexedCorpus that is the merge of ``existing`` and the
            indexable subset of ``new_documents``. The original ``existing``
            corpus is not mutated.
        """
        if not new_documents:
            logger.debug("update() called with no new documents; returning existing")
            return existing

        existing_ids = {doc.doc_id for doc in existing.documents}

        # Deep-ish copy of state we'll mutate
        kept_documents: list[Document] = list(existing.documents)
        kept_processed_texts: list[str] = list(existing.processed_texts)
        inverted_index: dict[str, list[tuple[int, int]]] = {
            term: list(postings) for term, postings in existing.inverted_index.items()
        }

        added = 0
        skipped_duplicate = 0
        skipped_short = 0

        for position, doc in enumerate(new_documents, start=1):
            if doc.doc_id in existing_ids:
                skipped_duplicate += 1
                continue

            processed = self.text_processor.process(doc.text, is_query=False)
            tokens = processed.split() if processed else []

            if len(tokens) < self.config.min_document_length:
                skipped_short += 1
                continue

            new_doc_idx = len(kept_documents)
            kept_documents.append(doc)
            kept_processed_texts.append(processed)
            existing_ids.add(doc.doc_id)

            for term, tf in Counter(tokens).items():
                inverted_index.setdefault(term, []).append((new_doc_idx, tf))

            added += 1

            if position % self.config.log_progress_every == 0:
                logger.info(
                    "update: processed %d/%d candidates", position, len(new_documents),
                )

        vocabulary = sorted(inverted_index.keys())

        logger.info(
            "update complete: +%d added, %d duplicates skipped, %d too short, "
            "total docs=%d, vocab=%d",
            added, skipped_duplicate, skipped_short,
            len(kept_documents), len(vocabulary),
        )

        return IndexedCorpus(
            documents=kept_documents,
            processed_texts=kept_processed_texts,
            inverted_index=inverted_index,
            vocabulary=vocabulary,
        )

    def remove(
        self,
        existing: IndexedCorpus,
        doc_ids: list[str],
    ) -> IndexedCorpus:
        """Remove documents from the corpus by doc_id.

        Produces a new IndexedCorpus in which the surviving documents are
        renumbered to contiguous indices (0..n-1), the inverted index is
        rewritten with the remapped indices, and terms whose postings become
        empty are dropped from the vocabulary.

        Args:
            existing: Current IndexedCorpus snapshot.
            doc_ids: IDs to remove. Unknown IDs are silently ignored.

        Returns:
            A new IndexedCorpus without the requested documents. The original
            is not mutated. Returns ``existing`` unchanged if ``doc_ids`` is
            empty.
        """
        if not doc_ids:
            return existing

        remove_set = set(doc_ids)

        # Build surviving docs and old_idx → new_idx remap in a single pass
        kept_documents: list[Document] = []
        kept_processed_texts: list[str] = []
        old_to_new: dict[int, int] = {}

        for old_idx, doc in enumerate(existing.documents):
            if doc.doc_id in remove_set:
                continue
            old_to_new[old_idx] = len(kept_documents)
            kept_documents.append(doc)
            kept_processed_texts.append(existing.processed_texts[old_idx])

        # Rewrite postings with remapped indices; drop terms that vanish
        inverted_index: dict[str, list[tuple[int, int]]] = {}
        for term, postings in existing.inverted_index.items():
            survivors = [
                (old_to_new[old_idx], tf)
                for old_idx, tf in postings
                if old_idx in old_to_new
            ]
            if survivors:
                inverted_index[term] = survivors

        vocabulary = sorted(inverted_index.keys())

        removed = len(existing.documents) - len(kept_documents)
        dropped_terms = len(existing.vocabulary) - len(vocabulary)
        logger.info(
            "remove complete: -%d docs (ignored %d unknown), -%d terms, "
            "total docs=%d, vocab=%d",
            removed,
            len(remove_set) - removed,
            dropped_terms,
            len(kept_documents),
            len(vocabulary),
        )

        return IndexedCorpus(
            documents=kept_documents,
            processed_texts=kept_processed_texts,
            inverted_index=inverted_index,
            vocabulary=vocabulary,
        )

    # ------------------------------------------------------------------
    # Management / inspection
    # ------------------------------------------------------------------

    @staticmethod
    def stats(corpus: IndexedCorpus) -> dict[str, float | int]:
        """Compute diagnostic statistics for an IndexedCorpus.

        Provides the numbers most often needed to monitor index health
        (size, average doc length, postings density). Cheap to compute —
        safe to call on every update for logging.

        Args:
            corpus: IndexedCorpus to inspect.

        Returns:
            Dict with ``n_documents``, ``n_terms``, ``total_tokens``,
            ``avg_tokens_per_doc``, ``avg_postings_per_term``.
        """
        n_docs = len(corpus.documents)
        n_terms = len(corpus.vocabulary)

        if n_docs == 0:
            return {
                "n_documents": 0,
                "n_terms": 0,
                "total_tokens": 0,
                "avg_tokens_per_doc": 0.0,
                "avg_postings_per_term": 0.0,
            }

        total_tokens = sum(len(text.split()) for text in corpus.processed_texts)
        total_postings = sum(
            len(postings) for postings in corpus.inverted_index.values()
        )

        return {
            "n_documents": n_docs,
            "n_terms": n_terms,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": total_tokens / n_docs,
            "avg_postings_per_term": (
                total_postings / n_terms if n_terms else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Query path
    # ------------------------------------------------------------------

    def build_query(self, query_text: str) -> IndexedCorpus:
        """Build a single-document IndexedCorpus from a user query.

        Uses ``is_query=True`` so TextProcessor applies spell correction
        against the vocabulary previously accumulated during build(). The
        downstream TfidfProcessor.transform() consumes the resulting
        inverted index and silently drops any term not in the document
        vocabulary.

        The IndexedCorpus invariant requires the documents list to match
        processed_texts in length, so a synthetic Document wrapping the
        raw query text is used as a placeholder.

        Args:
            query_text: Raw query string from the user. May contain typos.

        Returns:
            IndexedCorpus with exactly one document representing the query.
            Empty/whitespace input yields a valid empty query corpus.
        """
        if not query_text or not query_text.strip():
            logger.warning("build_query() called with empty query text")
            return IndexedCorpus(
                documents=[self._make_query_document(query_text)],
                processed_texts=[""],
                inverted_index={},
                vocabulary=[],
            )

        processed = self.text_processor.process(query_text, is_query=True)
        tokens = processed.split() if processed else []

        tf_counter = Counter(tokens)
        inverted_index: dict[str, list[tuple[int, int]]] = {
            term: [(0, tf)] for term, tf in tf_counter.items()
        }
        vocabulary = sorted(inverted_index.keys())

        logger.debug(
            "Built query corpus: raw=%r, tokens=%d, unique_terms=%d",
            query_text, len(tokens), len(vocabulary),
        )

        return IndexedCorpus(
            documents=[self._make_query_document(query_text)],
            processed_texts=[processed],
            inverted_index=inverted_index,
            vocabulary=vocabulary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate_postings(
        self,
        per_doc_tf: list[Counter[str]],
    ) -> tuple[dict[str, list[tuple[int, int]]], list[str]]:
        """Merge per-document term counts into an inverted index + vocabulary.

        Applies the ``min_term_frequency`` filter on the corpus-level total
        frequency of each term before returning. The vocabulary is sorted
        so that TF-IDF matrix columns are deterministic across runs.

        Args:
            per_doc_tf: One Counter per kept document, in corpus order.

        Returns:
            A pair ``(inverted_index, vocabulary)``. ``inverted_index`` maps
            term → list of ``(doc_idx, term_frequency)`` sorted by doc_idx
            (natural order of iteration). ``vocabulary`` is the sorted list
            of surviving terms.
        """
        inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total_term_freq: Counter[str] = Counter()

        for doc_idx, tf_counter in enumerate(per_doc_tf):
            for term, tf in tf_counter.items():
                inverted_index[term].append((doc_idx, tf))
                total_term_freq[term] += tf

        min_tf = self.config.min_term_frequency
        if min_tf > 1:
            dropped = [
                term for term, total in total_term_freq.items() if total < min_tf
            ]
            for term in dropped:
                inverted_index.pop(term, None)
            if dropped:
                logger.info(
                    "Dropped %d rare terms (total_freq < %d)", len(dropped), min_tf,
                )

        vocabulary = sorted(inverted_index.keys())
        return dict(inverted_index), vocabulary

    @staticmethod
    def _make_query_document(query_text: str) -> Document:
        """Create a synthetic Document to satisfy the IndexedCorpus invariant.

        The TF-IDF transform path never reads this document (it only uses
        inverted_index), but IndexedCorpus validates that documents and
        processed_texts have matching lengths.

        Args:
            query_text: Raw query text kept for traceability in metadata.

        Returns:
            Placeholder Document with a reserved doc_id and is_query marker.
        """
        return Document(
            doc_id=_QUERY_DOC_ID,
            text=query_text,
            url="",
            metadata={"is_query": True},
        )
