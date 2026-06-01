"""Rocchio relevance-feedback formula.

Classic algorithm from Rocchio (1971), used in vector-space IR for
explicit relevance feedback. Given an original query vector ``q`` and two
sets of judged documents (relevant ``D⁺`` and non-relevant ``D⁻``),
constructs a re-weighted query::

    q_new = α · q  +  β · (1/|D⁺|) · Σ d⁺  −  γ · (1/|D⁻|) · Σ d⁻

Conceptually it pulls the query toward the centroid of "documents the user
liked" and pushes it away from the centroid of "documents the user
disliked", with the original query held by an anchor weight ``α``.

The standard parameter choice (α=1.0, β=0.75, γ=0.15) is the one
documented by Manning, Raghavan & Schütze, *Introduction to Information
Retrieval* (Ch. 9), and is used as the default here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RocchioReweighter:
    """Pure-Python Rocchio re-weighting.

    Operates on plain ``list[float]`` vectors so the same instance works
    for both LSI latent vectors and raw TF-IDF vectors (caller chooses).

    Attributes:
        alpha: Weight of the original query vector. ``1.0`` preserves the
            user's intent as the anchor.
        beta: Weight applied to the *relevant* centroid. Higher values pull
            the query harder toward documents the user liked.
        gamma: Weight applied to the *non-relevant* centroid. Higher values
            push the query harder away from disliked documents. Often kept
            low because non-relevant feedback is noisy.
    """

    alpha: float = 1.0
    beta: float = 0.75
    gamma: float = 0.15

    def reweight(
        self,
        query_vector: list[float],
        relevant_vectors: list[list[float]],
        non_relevant_vectors: list[list[float]],
    ) -> list[float]:
        """Return the Rocchio-reweighted query vector.

        Args:
            query_vector: Original query embedding.
            relevant_vectors: Embeddings of documents marked relevant.
                Empty list is allowed — the β term becomes zero.
            non_relevant_vectors: Embeddings of documents marked
                non-relevant. Empty list is allowed — the γ term becomes
                zero. When both feedback lists are empty the original
                query is returned unchanged.

        Returns:
            New query vector of the same dimension as the input.

        Raises:
            ValueError: If any input vector has a different dimension from
                ``query_vector``.
        """
        if not query_vector:
            raise ValueError("query_vector must not be empty")

        dim = len(query_vector)
        self._validate_dimension(dim, relevant_vectors, "relevant_vectors")
        self._validate_dimension(dim, non_relevant_vectors, "non_relevant_vectors")

        if not relevant_vectors and not non_relevant_vectors:
            # Nothing to add or subtract — return a copy so callers don't
            # accidentally share aliased state with the input.
            return list(query_vector)

        rel_centroid = self._centroid(relevant_vectors, dim)
        nonrel_centroid = self._centroid(non_relevant_vectors, dim)

        return [
            self.alpha * q + self.beta * r - self.gamma * n
            for q, r, n in zip(query_vector, rel_centroid, nonrel_centroid)
        ]

    @staticmethod
    def _centroid(vectors: list[list[float]], dim: int) -> list[float]:
        """Mean of ``vectors`` component-wise, or a zero vector if empty."""
        if not vectors:
            return [0.0] * dim
        n = len(vectors)
        sums = [0.0] * dim
        for vec in vectors:
            for i, v in enumerate(vec):
                sums[i] += v
        return [s / n for s in sums]

    @staticmethod
    def _validate_dimension(
        expected: int, vectors: list[list[float]], label: str
    ) -> None:
        for i, vec in enumerate(vectors):
            if len(vec) != expected:
                raise ValueError(
                    f"{label}[{i}] has dimension {len(vec)}, expected {expected}"
                )
