"""Stores for relevance judgments — in-memory and JSONL-on-disk.

Both expose the same minimal contract so the high-level
:class:`~plugins.feedback.service.RelevanceFeedbackService` is agnostic to
where the data lives.

Persistence format (JSONL): one judgment per line, UTF-8 encoded::

    {"query_text": "diabetes", "doc_id": "d1", "relevant": true, "timestamp": "..."}

The in-memory variant is used for tests and ephemeral sessions; the
file-backed variant is what production-style usage from the UI would
plug in.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Protocol

from .models import RelevanceJudgment

logger = logging.getLogger(__name__)


class FeedbackStore(Protocol):
    """Read-write store for relevance judgments."""

    def add(self, judgment: RelevanceJudgment) -> None:
        """Append a judgment to the store."""
        ...

    def get_for_query(
        self, query_text: str
    ) -> tuple[list[str], list[str]]:
        """Return ``(relevant_ids, non_relevant_ids)`` for ``query_text``."""
        ...

    def clear(self) -> None:
        """Wipe every recorded judgment."""
        ...

    def __len__(self) -> int:
        ...


class InMemoryFeedbackStore:
    """List-backed feedback store. Fast; not persistent.

    Two passes over the in-memory list are not a concern for the volumes
    realistically produced by user feedback — a few hundred judgments per
    user session at most.
    """

    def __init__(self) -> None:
        self._judgments: list[RelevanceJudgment] = []

    def add(self, judgment: RelevanceJudgment) -> None:
        # Replace any prior judgment for the same (query, doc) pair so
        # the user's *latest* opinion always wins. Identity is the
        # normalised query plus doc_id.
        key = (
            RelevanceJudgment.normalise_query(judgment.query_text),
            judgment.doc_id,
        )
        self._judgments = [
            j for j in self._judgments
            if (RelevanceJudgment.normalise_query(j.query_text), j.doc_id) != key
        ]
        self._judgments.append(judgment)

    def get_for_query(
        self, query_text: str
    ) -> tuple[list[str], list[str]]:
        key = RelevanceJudgment.normalise_query(query_text)
        relevant: list[str] = []
        non_relevant: list[str] = []
        for j in self._judgments:
            if RelevanceJudgment.normalise_query(j.query_text) != key:
                continue
            (relevant if j.relevant else non_relevant).append(j.doc_id)
        return relevant, non_relevant

    def clear(self) -> None:
        self._judgments.clear()

    def all_judgments(self) -> list[RelevanceJudgment]:
        """Return a copy of every stored judgment (insertion order)."""
        return list(self._judgments)

    def __len__(self) -> int:
        return len(self._judgments)


class JSONLFeedbackStore:
    """File-backed feedback store using newline-delimited JSON.

    The on-disk format is append-only: each call to :meth:`add` opens the
    file in append mode and writes one record. Lookups stream the file end
    to end; for the volumes user feedback produces this stays fast.

    Concurrent writers are not supported (single-process tool); the file
    is opened only for the duration of each operation so multiple
    *instances* in the same process can coexist.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add(self, judgment: RelevanceJudgment) -> None:
        record = {
            "query_text": judgment.query_text,
            "doc_id": judgment.doc_id,
            "relevant": judgment.relevant,
            "timestamp": judgment.timestamp,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_for_query(
        self, query_text: str
    ) -> tuple[list[str], list[str]]:
        key = RelevanceJudgment.normalise_query(query_text)
        # Collect latest opinion per (query, doc_id) — later writes win.
        latest: dict[str, bool] = {}
        for j in self._iter_all():
            if RelevanceJudgment.normalise_query(j.query_text) != key:
                continue
            latest[j.doc_id] = j.relevant

        relevant = [d for d, r in latest.items() if r]
        non_relevant = [d for d, r in latest.items() if not r]
        return relevant, non_relevant

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def all_judgments(self) -> list[RelevanceJudgment]:
        return list(self._iter_all())

    def __len__(self) -> int:
        return sum(1 for _ in self._iter_all())

    def _iter_all(self) -> Iterable[RelevanceJudgment]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "JSONLFeedbackStore: skipping malformed line in %s: %s",
                        self.path, exc,
                    )
                    continue
                yield RelevanceJudgment(
                    query_text=record["query_text"],
                    doc_id=record["doc_id"],
                    relevant=bool(record["relevant"]),
                    timestamp=record.get("timestamp", ""),
                )
