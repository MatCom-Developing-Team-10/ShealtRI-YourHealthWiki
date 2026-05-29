"""Data models for the relevance-feedback plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RelevanceJudgment:
    """A single ``(query, doc, label)`` judgment supplied by a user.

    Attributes:
        query_text: Normalised query string the judgment was given for. The
            service caller is responsible for normalisation (lower-case,
            strip) so that two semantically identical queries share their
            feedback bucket.
        doc_id: Identifier of the judged document.
        relevant: ``True`` for relevant / "thumbs up", ``False`` for
            irrelevant / "thumbs down".
        timestamp: ISO-8601 UTC timestamp of when the judgment was made.
            Defaults to ``datetime.now(timezone.utc)`` at construction.
    """

    query_text: str
    doc_id: str
    relevant: bool
    timestamp: str = field(default_factory=_utcnow_iso)

    @staticmethod
    def normalise_query(text: str) -> str:
        """Canonical form used to key judgments — single source of truth.

        Lower-cases, strips, and collapses internal whitespace. Two queries
        that differ only in casing or spacing are considered the same.
        """
        return " ".join(text.lower().split())
