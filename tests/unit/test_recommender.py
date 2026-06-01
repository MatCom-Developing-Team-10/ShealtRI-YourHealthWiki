"""Unit tests for ContentBasedRecommender.

Uses the in-memory fakes from ``conftest`` so the recommender's logic is
exercised without ChromaDB or LSI fitting overhead. Latent vectors are
hand-crafted so the cosine distances are predictable.
"""

from __future__ import annotations

import math

import pytest

from core.models import Document, UserProfileType
from modules.recommender import ContentBasedRecommender, RecommendedDocument
from tests.conftest import InMemoryDocumentStore, InMemoryRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(doc_id: str, source: str = "medlineplus", original: str | None = None) -> Document:
    meta = {"source": source, "title": doc_id.replace("_", " ").title()}
    if original is not None:
        meta["original_doc_id"] = original
    return Document(
        doc_id=doc_id,
        text=f"contenido del documento {doc_id}",
        url=f"https://example.org/{doc_id}",
        metadata=meta,
    )


def _populate(store: InMemoryDocumentStore, repo: InMemoryRepository,
              entries: list[tuple[Document, list[float]]]) -> None:
    """Add documents + embeddings to both stores in one shot."""
    docs = [doc for doc, _ in entries]
    vecs = [vec for _, vec in entries]
    store.add_documents(docs)
    repo.add_documents(docs, embeddings=vecs)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_lambda(self, in_memory_repo, in_memory_store):
        rec = ContentBasedRecommender(in_memory_repo, in_memory_store)
        assert rec.mmr_lambda == 0.7

    def test_rejects_lambda_out_of_range(self, in_memory_repo, in_memory_store):
        with pytest.raises(ValueError):
            ContentBasedRecommender(in_memory_repo, in_memory_store, mmr_lambda=1.5)
        with pytest.raises(ValueError):
            ContentBasedRecommender(in_memory_repo, in_memory_store, mmr_lambda=-0.1)

    def test_rejects_zero_pool(self, in_memory_repo, in_memory_store):
        with pytest.raises(ValueError):
            ContentBasedRecommender(
                in_memory_repo, in_memory_store, candidate_pool_size=0,
            )


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_no_seeds_returns_empty(self, in_memory_repo, in_memory_store):
        rec = ContentBasedRecommender(in_memory_repo, in_memory_store)
        assert rec.recommend([]) == []

    def test_top_k_zero_returns_empty(self, in_memory_repo, in_memory_store):
        rec = ContentBasedRecommender(in_memory_repo, in_memory_store)
        assert rec.recommend(["d1"], top_k=0) == []

    def test_unknown_seeds_returns_empty(self, in_memory_repo, in_memory_store):
        """If no seed has an embedding, the centroid is undefined → empty."""
        rec = ContentBasedRecommender(in_memory_repo, in_memory_store)
        assert rec.recommend(["nonexistent"]) == []


# ---------------------------------------------------------------------------
# Basic recommendation behaviour
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_returns_most_similar_excluding_seeds(self, in_memory_repo, in_memory_store):
        # Seed is identical to itself; d2 is the next closest.
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),       [1.0, 0.0, 0.0]),
            (_make_doc("d2"),         [0.95, 0.1, 0.0]),
            (_make_doc("d3"),         [0.0, 1.0, 0.0]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=2)
        ids = [r.document.doc_id for r in out]
        assert "seed" not in ids
        assert ids[0] == "d2"

    def test_respects_top_k(self, in_memory_repo, in_memory_store):
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"), [1.0, 0.0]),
            (_make_doc("d1"),   [0.9, 0.1]),
            (_make_doc("d2"),   [0.8, 0.2]),
            (_make_doc("d3"),   [0.7, 0.3]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=2)
        assert len(out) == 2

    def test_respects_exclude_ids(self, in_memory_repo, in_memory_store):
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),    [1.0, 0.0]),
            (_make_doc("already"), [0.95, 0.05]),
            (_make_doc("new"),     [0.90, 0.10]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=2, exclude_ids={"already"})
        ids = [r.document.doc_id for r in out]
        assert "already" not in ids
        assert "new" in ids

    def test_drops_candidates_below_min_similarity(
        self, in_memory_repo, in_memory_store,
    ):
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),  [1.0, 0.0]),
            (_make_doc("close"), [0.9, 0.1]),    # sim ~1.0
            (_make_doc("far"),   [-1.0, 0.0]),   # sim = -1 < 0.3
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.30,
        )
        out = rec.recommend(["seed"], top_k=5)
        ids = [r.document.doc_id for r in out]
        assert "far" not in ids

    def test_deduplicates_chunks_of_same_original(
        self, in_memory_repo, in_memory_store,
    ):
        """Two chunks of the same parent document should collapse to one rec."""
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),                                 [1.0, 0.0]),
            (_make_doc("docA__chunk_0", original="docA"),       [0.95, 0.05]),
            (_make_doc("docA__chunk_1", original="docA"),       [0.93, 0.07]),
            (_make_doc("docB__chunk_0", original="docB"),       [0.80, 0.20]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=5)
        # Only the highest-scoring chunk of docA survives, plus docB.
        originals = {
            r.document.metadata.get("original_doc_id", r.document.doc_id)
            for r in out
        }
        assert originals == {"docA", "docB"}

    def test_deduplicates_pages_of_same_book(
        self, in_memory_repo, in_memory_store,
    ):
        """Three pages of the same book collapse to one recommendation.

        The PDF corpus is ingested one page per "original document"
        (``book_pNNN``). Without book-level dedup the user would see
        three near-identical recommendations from the same textbook.
        """
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed",                original="seed"),                [1.0, 0.0]),
            (_make_doc("Guyton_p100__chunk_0", original="Guyton_p100"),       [0.95, 0.05]),
            (_make_doc("Guyton_p101__chunk_0", original="Guyton_p101"),       [0.93, 0.07]),
            (_make_doc("Guyton_p102__chunk_0", original="Guyton_p102"),       [0.92, 0.08]),
            (_make_doc("Saladin_p50__chunk_0", original="Saladin_p50"),       [0.80, 0.20]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=5)
        # Only the best page of Guyton survives, plus Saladin.
        originals = [
            r.document.metadata.get("original_doc_id", r.document.doc_id)
            for r in out
        ]
        assert "Guyton_p100" in originals
        assert "Guyton_p101" not in originals
        assert "Guyton_p102" not in originals
        assert "Saladin_p50" in originals

    def test_web_docs_without_page_suffix_remain_distinct(
        self, in_memory_repo, in_memory_store,
    ):
        """The book-dedup regex must not collapse unrelated web pages."""
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed", original="seed"),                                          [1.0, 0.0]),
            (_make_doc("medlineplus_diabetes_overview", original="medlineplus_diabetes_overview"), [0.95, 0.05]),
            (_make_doc("medlineplus_diabetes_symptoms", original="medlineplus_diabetes_symptoms"), [0.93, 0.07]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=5)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Centroid averaging
# ---------------------------------------------------------------------------


class TestCentroid:
    def test_centroid_is_arithmetic_mean(
        self, in_memory_repo, in_memory_store,
    ):
        """Two seeds at [1,0] and [0,1] should pull toward a candidate near [0.5, 0.5]."""
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("s1"),       [1.0, 0.0]),
            (_make_doc("s2"),       [0.0, 1.0]),
            (_make_doc("midpoint"), [1.0, 1.0]),  # cosine to centroid = 1
            (_make_doc("offaxis"),  [1.0, -1.0]), # cosine to centroid = 0
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=-1.0,
        )
        out = rec.recommend(["s1", "s2"], top_k=1)
        assert out[0].document.doc_id == "midpoint"


# ---------------------------------------------------------------------------
# MMR diversification
# ---------------------------------------------------------------------------


class TestMMR:
    def test_lambda_one_is_pure_similarity(self, in_memory_repo, in_memory_store):
        """With λ=1, MMR collapses to pure cosine — twin candidates both appear."""
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),  [1.0, 0.0]),
            (_make_doc("twinA"), [0.95, 0.05]),
            (_make_doc("twinB"), [0.94, 0.06]),  # near-duplicate of twinA
            (_make_doc("other"), [0.50, 0.50]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store,
            mmr_lambda=1.0, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=2)
        ids = [r.document.doc_id for r in out]
        # Both twins beat 'other' on raw similarity.
        assert set(ids) == {"twinA", "twinB"}

    def test_lambda_low_prefers_diversity(self, in_memory_repo, in_memory_store):
        """With low λ the second pick switches away from a near-duplicate."""
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),  [1.0, 0.0]),
            (_make_doc("twinA"), [0.95, 0.05]),
            (_make_doc("twinB"), [0.94, 0.06]),
            (_make_doc("other"), [0.50, 0.50]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store,
            mmr_lambda=0.1, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=2)
        ids = [r.document.doc_id for r in out]
        # First pick is twinA (highest sim). Second pick must NOT be twinB
        # (high redundancy) → it must be 'other'.
        assert ids[0] in {"twinA", "twinB"}
        assert "other" in ids

    def test_first_pick_mmr_equals_similarity(self, in_memory_repo, in_memory_store):
        """For the first selection there is no picked set yet → mmr == sim."""
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"), [1.0, 0.0]),
            (_make_doc("d1"),   [0.9, 0.1]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store,
            mmr_lambda=0.5, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=1)
        assert math.isclose(out[0].mmr_score, out[0].similarity, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Profile boost
# ---------------------------------------------------------------------------


class TestProfileBoost:
    def test_patient_profile_boosts_medlineplus(self, in_memory_repo, in_memory_store):
        """Two near-tied candidates: medlineplus must win for PATIENT."""
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),                            [1.0, 0.0]),
            (_make_doc("technical", source="nhs"),         [0.90, 0.10]),
            (_make_doc("lay", source="medlineplus"),       [0.88, 0.12]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=1, profile=UserProfileType.PATIENT)
        assert out[0].document.doc_id == "lay"

    def test_professional_profile_boosts_nhs(self, in_memory_repo, in_memory_store):
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),                            [1.0, 0.0]),
            (_make_doc("technical", source="nhs"),         [0.88, 0.12]),
            (_make_doc("lay", source="medlineplus"),       [0.90, 0.10]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(
            ["seed"], top_k=1, profile=UserProfileType.MEDICAL_PROFESSIONAL,
        )
        assert out[0].document.doc_id == "technical"

    def test_natural_profile_is_neutral(self, in_memory_repo, in_memory_store):
        """No boost map for NATURAL_MEDICINE → pure similarity wins."""
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"),                            [1.0, 0.0]),
            (_make_doc("d_nhs", source="nhs"),             [0.85, 0.15]),
            (_make_doc("d_medline", source="medlineplus"), [0.95, 0.05]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(
            ["seed"], top_k=1, profile=UserProfileType.NATURAL_MEDICINE,
        )
        assert out[0].document.doc_id == "d_medline"


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_returns_recommended_documents(self, in_memory_repo, in_memory_store):
        _populate(in_memory_store, in_memory_repo, [
            (_make_doc("seed"), [1.0, 0.0]),
            (_make_doc("d1"),   [0.9, 0.1]),
        ])
        rec = ContentBasedRecommender(
            in_memory_repo, in_memory_store, min_similarity=0.0,
        )
        out = rec.recommend(["seed"], top_k=1)
        assert len(out) == 1
        assert isinstance(out[0], RecommendedDocument)
        assert isinstance(out[0].document, Document)
        assert 0.0 <= out[0].similarity <= 1.0 + 1e-9
