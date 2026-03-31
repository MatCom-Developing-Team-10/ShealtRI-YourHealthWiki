"""Shared domain models for the SRI pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Query:
    """Represents an incoming user query.

    Attributes:
        text: Raw query string entered by the user.
        metadata: Optional metadata associated with the query.
    """

    text: str
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
