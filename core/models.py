"""Shared domain models for the SRI pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.interfaces import IndexedCorpus


@dataclass(slots=True)
class Query:
    """Represents an incoming user query.

    Attributes:
        text: Raw query string entered by the user.
        indexed_corpus: Preprocessed query as IndexedCorpus for retrieval.
            Built by the pipeline before calling the retriever.
        metadata: Optional metadata associated with the query.
    """

    text: str
    indexed_corpus: IndexedCorpus | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """Represents a retrievable document in the corpus.

    Attributes:
        doc_id: Unique identifier for a document.
        text: Plain text content used by retrievers.
        metadata: Optional metadata (source URL, title, tags, etc.).
    """

    doc_id: str
    text: str
    url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedDocument:
    """Represents a retrieved document and its relevance score."""

    document: Document
    score: float


@dataclass(slots=True)
class PipelineContext:
    """Mutable context object passed between pipeline stages and plugins.

    The pipeline (or any orchestrator) creates a context, hands it to each
    pre_retrieval plugin, calls the retriever, hands it to each
    post_retrieval / post_ranking plugin, and returns ``results`` at the end.

    Plugins should mutate the context in place (or return a new instance);
    by convention they leave a breadcrumb under ``metadata[plugin_name]``
    so debugging the pipeline does not require reading every plugin's source.

    Attributes:
        query: The user query with ``indexed_corpus`` populated by the indexer.
        results: Documents produced by the retriever (empty before retrieval).
        metadata: Free-form storage used by plugins to record what they did.
    """

    query: Query
    results: list[RetrievedDocument] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
