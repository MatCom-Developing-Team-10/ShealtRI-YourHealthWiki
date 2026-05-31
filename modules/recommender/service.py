"""Content-based recommender — LSI similarity with MMR diversification.

The recommender takes a *seed* set of doc_ids (typically the top-N results
of the user's latest search), averages their latent vectors to form a
centroid in the LSI space, retrieves candidate documents by cosine
similarity, and re-orders them with Maximal Marginal Relevance (MMR) so
the final list is similar to the seed but internally diverse.

Crucially, the recommender does NOT touch the ranking of the main query.
It is exposed as a separate endpoint (``/api/recommend``) and feeds a
separate UI panel ("También te puede interesar"), so it is orthogonal to
Rocchio re-weighting, the HybridRanker, and the evaluation dataset.

MMR formula (Carbonell & Goldstein, SIGIR 1998):

    MMR(d) = λ · sim(d, centroid) − (1 − λ) · max_{p ∈ picked} sim(d, p)

With ``λ = 1.0`` it degenerates to pure cosine similarity.  ``λ = 0.0``
picks only on inter-document dissimilarity.  The default 0.7 favours
similarity to the seed while avoiding near-duplicates.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import numpy as np

# Books in the academic corpus are ingested one page per "document", so the
# original_doc_id looks like ``<book>_p<NNN>``. Stripping the page suffix
# gives a stable book identifier, used to avoid showing three near-identical
# pages of the same textbook as separate recommendations.
_BOOK_FROM_PAGE = re.compile(r"^(.*?)_p\d+$")


def _book_key(original_doc_id: str) -> str:
    """Collapse a per-page original_doc_id to its parent book identifier.

    Falls back to the input untouched when the id does not match the
    page-suffix convention (e.g. web-crawled docs).
    """
    match = _BOOK_FROM_PAGE.match(original_doc_id)
    return match.group(1) if match else original_doc_id

from core.interfaces import BaseRepository, DocumentStore
from core.models import Document, UserProfileType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile → preferred source bucket. Boost is multiplicative on similarity.
# Boost values are deliberately conservative (1.05–1.10) so the recommender
# stays *similarity-first*; the profile only nudges ties.
# ---------------------------------------------------------------------------

_PROFILE_SOURCE_BOOST: dict[UserProfileType, dict[str, float]] = {
    UserProfileType.PATIENT:              {"medlineplus": 1.10},
    UserProfileType.CAREGIVER:            {"medlineplus": 1.10},
    UserProfileType.NATURAL_MEDICINE:     {},
    UserProfileType.MEDICAL_STUDENT:      {"mayo_clinic": 1.05, "nhs": 1.05},
    UserProfileType.MEDICAL_PROFESSIONAL: {"mayo_clinic": 1.10, "nhs": 1.10},
    UserProfileType.DIAGNOSTIC_ASSISTANT: {"mayo_clinic": 1.10, "nhs": 1.10},
}


@dataclass(slots=True)
class RecommendedDocument:
    """One recommendation, with both the raw similarity and the final MMR score.

    Attributes:
        document: Full document to display.
        similarity: Cosine similarity to the seed centroid in [0, 1].
            Boost-adjusted when a profile is provided.
        mmr_score: Score the recommender used to pick this item over the
            remaining candidates. Equal to ``similarity`` for the first
            pick; lower for later picks (the redundancy penalty).
    """

    document: Document
    similarity: float
    mmr_score: float


class ContentBasedRecommender:
    """LSI-similarity recommender with MMR diversification.

    Args:
        repository: Vector store exposing ``get_embedding`` and
            ``search_similar`` — typically the same Chroma repository the
            LSI retriever uses.
        document_store: Source of full :class:`Document` objects for the
            ids returned by the repository.
        mmr_lambda: MMR trade-off in [0, 1]. Higher = more similar to
            seed, lower = more diverse. Default 0.7.
        candidate_pool_size: How many candidates to fetch from the vector
            store before MMR re-ranking. Larger pools give MMR more room
            to diversify, at the cost of slightly more cosine work.
        min_similarity: Minimum base cosine similarity for a candidate to
            survive filtering. Avoids returning obviously off-topic docs
            when the corpus is small or the seed is niche.
    """

    def __init__(
        self,
        repository: BaseRepository,
        document_store: DocumentStore,
        mmr_lambda: float = 0.7,
        candidate_pool_size: int = 30,
        min_similarity: float = 0.30,
    ) -> None:
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be >= 1")
        self.repository = repository
        self.document_store = document_store
        self.mmr_lambda = mmr_lambda
        self.candidate_pool_size = candidate_pool_size
        self.min_similarity = min_similarity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        seed_doc_ids: list[str],
        top_k: int = 3,
        exclude_ids: set[str] | None = None,
        profile: UserProfileType | None = None,
    ) -> list[RecommendedDocument]:
        """Return up to ``top_k`` recommendations.

        Args:
            seed_doc_ids: Document (or chunk) ids whose latent vectors
                seed the recommendation. The list of top results from the
                user's last query is a natural choice.
            top_k: How many recommendations to return.
            exclude_ids: Ids that must not appear in the output. The seed
                ids themselves are always excluded automatically. Use this
                to also exclude the ids the user has already seen in the
                main result list.
            profile: When provided, apply the small profile-based source
                boost (see :data:`_PROFILE_SOURCE_BOOST`).

        Returns:
            Up to ``top_k`` :class:`RecommendedDocument` objects, ordered
            by descending MMR score. Empty list if seeds yielded no
            candidates above ``min_similarity`` or no embeddings were
            available for any seed.
        """
        if not seed_doc_ids or top_k <= 0:
            return []

        exclude = set(exclude_ids or set())
        exclude.update(seed_doc_ids)

        centroid = self._seed_centroid(seed_doc_ids)
        if centroid is None:
            logger.debug(
                "ContentBasedRecommender: no embeddings found for seeds %s",
                seed_doc_ids,
            )
            return []

        # Over-fetch: ask for pool + seeds so excluding seeds still leaves a
        # full pool to work with.
        raw = self.repository.search_similar(
            centroid, top_k=self.candidate_pool_size + len(seed_doc_ids),
        )

        candidates = self._build_candidates(raw, exclude, profile)
        if not candidates:
            return []

        return self._mmr_select(candidates, top_k)

    # ------------------------------------------------------------------
    # Internal — candidate building
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        raw: list[tuple[str, float]],
        exclude: set[str],
        profile: UserProfileType | None,
    ) -> list[tuple[Document, list[float], float]]:
        """Filter, deduplicate by original document, and look up vectors.

        Two dedup levels are applied:

        1. Chunk → original_doc_id: discard secondary chunks of the same
           page, keeping only the highest-scoring one.
        2. original_doc_id → book: PDF books are ingested one page per
           "original document", so without this step three consecutive
           pages of the same textbook would surface as three separate
           recommendations. ``_book_key`` collapses them.

        Web-crawled docs do not match the ``..._pNNN`` convention, so
        ``_book_key`` is a no-op for them — each web doc remains its own
        "book".
        """
        boost_map = _PROFILE_SOURCE_BOOST.get(profile, {}) if profile else {}
        candidates: list[tuple[Document, list[float], float]] = []
        seen_originals: set[str] = set()
        seen_books: set[str] = set()

        for chunk_id, score in raw:
            if chunk_id in exclude or score < self.min_similarity:
                continue

            doc = self.document_store.get_by_id(chunk_id)
            if doc is None:
                continue

            original = doc.metadata.get(
                "original_doc_id", chunk_id.split("__chunk_")[0],
            )
            if original in seen_originals:
                continue

            book = _book_key(original)
            if book in seen_books:
                continue

            vector = self.repository.get_embedding(chunk_id)
            if vector is None:
                continue

            source = doc.metadata.get("source", "")
            adjusted = score * boost_map.get(source, 1.0)

            seen_originals.add(original)
            seen_books.add(book)
            candidates.append((doc, vector, adjusted))

        # Re-sort because the boost may have changed the order.
        candidates.sort(key=lambda c: c[2], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Internal — MMR re-ranking
    # ------------------------------------------------------------------

    def _mmr_select(
        self,
        candidates: list[tuple[Document, list[float], float]],
        top_k: int,
    ) -> list[RecommendedDocument]:
        """Greedy MMR selection of ``top_k`` items from ``candidates``."""
        picked: list[tuple[Document, list[float], float, float]] = []
        remaining = list(candidates)

        while remaining and len(picked) < top_k:
            best_idx, best_mmr = -1, -math.inf
            for i, (_, vec, sim) in enumerate(remaining):
                if not picked:
                    mmr = sim
                else:
                    redundancy = max(
                        self._cosine(vec, picked_vec)
                        for _, picked_vec, _, _ in picked
                    )
                    mmr = self.mmr_lambda * sim - (1 - self.mmr_lambda) * redundancy
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            doc, vec, sim = remaining.pop(best_idx)
            picked.append((doc, vec, sim, best_mmr))

        return [
            RecommendedDocument(document=doc, similarity=sim, mmr_score=mmr)
            for doc, _, sim, mmr in picked
        ]

    # ------------------------------------------------------------------
    # Internal — vector math
    # ------------------------------------------------------------------

    def _seed_centroid(self, seed_doc_ids: list[str]) -> list[float] | None:
        """Average the latent vectors of all seeds for which we have one."""
        vectors: list[list[float]] = []
        for doc_id in seed_doc_ids:
            vec = self.repository.get_embedding(doc_id)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return None
        arr = np.asarray(vectors, dtype=float)
        return arr.mean(axis=0).tolist()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        av = np.asarray(a, dtype=float)
        bv = np.asarray(b, dtype=float)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom == 0.0:
            return 0.0
        return float(np.dot(av, bv) / denom)
