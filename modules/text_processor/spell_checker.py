"""Trie-based spell checker for medical queries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, _TrieNode] = field(default_factory=dict)
    is_end: bool = False


class TrieSpellChecker:
    """Spell checker constrained by corpus vocabulary.

    Corrects misspelled words by finding the closest match in the vocabulary
    using Levenshtein distance. Used during query processing.
    """

    def __init__(self, vocabulary: list[str] | None = None, max_distance: int = 2) -> None:
        """Initialize with vocabulary.

        Args:
            vocabulary: List of valid words from the corpus.
            max_distance: Maximum edit distance for corrections (default: 2).
        """
        self.root = _TrieNode()
        self.max_distance = max_distance
        if vocabulary:
            self.fit(vocabulary)

    def fit(self, vocabulary: list[str]) -> None:
        """Populate the Trie with a list of valid words.

        Args:
            vocabulary: List of valid words to add to the Trie.
        """
        for word in vocabulary:
            self._insert(word)

    def _insert(self, word: str) -> None:
        """Add a word to the Trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = _TrieNode()
            node = node.children[char]
        node.is_end = True

    def _contains(self, word: str) -> bool:
        """Check if a word exists in the Trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(a) < len(b):
            return TrieSpellChecker._levenshtein(b, a)
        if len(b) == 0:
            return len(a)

        previous_row = range(len(b) + 1)
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def correct(self, word: str) -> str | None:
        """Return the closest known vocabulary word or None if not found.

        Args:
            word: Word to correct.

        Returns:
            The closest matching word from vocabulary, or None if no match
            within max_distance is found.
        """
        if self._contains(word):
            return word

        results = []
        self._search_recursive(self.root, "", word, results)

        if not results:
            return None

        # Return the closest match
        results.sort(key=lambda x: x[1])
        return results[0][0]

    def _search_recursive(
        self,
        node: _TrieNode,
        current_word: str,
        target: str,
        results: list[tuple[str, int]],
    ) -> None:
        """Recursively search Trie for words close to target."""
        dist = self._levenshtein(current_word, target)

        if dist <= self.max_distance and node.is_end:
            results.append((current_word, dist))

        # Prune branches that are too different
        if len(current_word) > len(target) + self.max_distance:
            return

        for char, child in node.children.items():
            self._search_recursive(child, current_word + char, target, results)
