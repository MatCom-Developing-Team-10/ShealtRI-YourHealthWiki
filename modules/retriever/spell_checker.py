"""Trie-based spell checker used before LSI query vectorization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class TrieSpellChecker:
    """Simple spell checker constrained by corpus vocabulary."""

    def __init__(self, vocabulary: list[str]) -> None:
        raise NotImplementedError

    def _insert(self, word: str) -> None:
        raise NotImplementedError

    def _contains(self, word: str) -> bool:
        raise NotImplementedError

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        raise NotImplementedError

    def correct_word(self, word: str, max_distance: int = 2) -> str:
        """Return the closest known vocabulary word or the original token."""
        raise NotImplementedError
