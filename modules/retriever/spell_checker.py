"""Trie-based spell checker used before LSI query vectorization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class TrieSpellChecker:
    """Simple spell checker constrained by corpus vocabulary."""

    def __init__(self, vocabulary: list[str] | None = None, max_distance: int = 2) -> None:
        """Initialize with vocabulary.

        Args:
            vocabulary: Initial list of words.
            max_distance: Max edit distance for correction.
        """
        self.root = _TrieNode()
        self.max_distance = max_distance
        if vocabulary:
            self.fit(vocabulary)

    def fit(self, vocabulary: list[str]) -> None:
        """Populate the Trie with a list of words."""
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

    def correct(self, word: str) -> str:
        """Return the closest known vocabulary word or the original token."""
        if self._contains(word):
            return word

        # Simple exhaustive search for now (could be optimized with Trie branch pruning)
        # For a small/medium medical vocabulary, this might be okay.
        best_word = word
        min_dist = self.max_distance + 1

        # This is a naive implementation. A better one would traverse the Trie.
        # But for now, let's keep it simple or implement the Trie traversal if needed.
        # Let's do a simple recursive Trie search with distance pruning.
        
        results = []
        self._search_recursive(self.root, "", word, results)
        
        if not results:
            return word
            
        # Return the one with minimum distance
        results.sort(key=lambda x: x[1])
        return results[0][0]

    def _search_recursive(self, node: _TrieNode, current_word: str, target: str, results: list[tuple[str, int]]):
        dist = self._levenshtein(current_word, target)
        
        if dist <= self.max_distance and node.is_end:
            results.append((current_word, dist))
            
        # Pruning: if current_word is already too different, we might still want to continue 
        # because adding/deleting chars might decrease distance. 
        # But usually, if len(current_word) > len(target) + max_distance, we can stop.
        if len(current_word) > len(target) + self.max_distance:
            return

        for char, child in node.children.items():
            self._search_recursive(child, current_word + char, target, results)
