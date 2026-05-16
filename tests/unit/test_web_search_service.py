"""Unit tests for the WebSearchRetriever module.

Tests keyword extraction, document scoring, and end-to-end retrieval.
"""

import pytest

from core.interfaces import DocumentStore
from core.models import Document, Query, RetrievedDocument
from modules.web_search.service import WebSearchRetriever


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


class MockDocumentStore(DocumentStore):
    """In-memory document store for testing."""

    def __init__(self, documents: dict[str, Document] | None = None) -> None:
        """Initialize with optional documents.

        Args:
            documents: Dict of doc_id -> Document.
        """
        self._store = documents or {}

    def add_documents(self, documents: list[Document]) -> None:
        """Store documents by ID."""
        for doc in documents:
            self._store[doc.doc_id] = doc

    def get_by_ids(self, doc_ids: list[str]) -> list[Document]:
        """Retrieve multiple documents by ID."""
        return [self._store[doc_id] for doc_id in doc_ids if doc_id in self._store]

    def get_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a single document by ID."""
        return self._store.get(doc_id)

    def list_all_ids(self) -> list[str]:
        """Return all document IDs."""
        return list(self._store.keys())


@pytest.fixture
def sample_documents() -> dict[str, Document]:
    """Create sample medical documents for testing."""
    return {
        "doc1": Document(
            doc_id="doc1",
            text="Hypertension is a condition where blood pressure is elevated. "
                 "Common treatments include ACE inhibitors and beta blockers. "
                 "Patients should monitor their blood pressure regularly.",
            url="https://example.com/hypertension",
            metadata={"title": "Understanding Hypertension", "source": "mayo_clinic"},
        ),
        "doc2": Document(
            doc_id="doc2",
            text="Diabetes mellitus is a metabolic disorder affecting glucose regulation. "
                 "Type 2 diabetes is the most common form. "
                 "Treatment may include insulin or oral medications.",
            url="https://example.com/diabetes",
            metadata={"title": "Diabetes Management Guide", "source": "medlineplus"},
        ),
        "doc3": Document(
            doc_id="doc3",
            text="Arthritis causes inflammation of the joints. "
                 "Symptoms include pain, swelling, and stiffness. "
                 "Treatment options range from physical therapy to medication.",
            url="https://example.com/arthritis",
            metadata={"title": "Arthritis Relief", "source": "nhs"},
        ),
    }


@pytest.fixture
def mock_store(sample_documents: dict[str, Document]) -> MockDocumentStore:
    """Create a mock document store with sample documents."""
    return MockDocumentStore(sample_documents)


@pytest.fixture
def retriever(mock_store: MockDocumentStore) -> WebSearchRetriever:
    """Create a WebSearchRetriever instance with mock store."""
    from core.stopwords import ENGLISH_STOPWORDS
    return WebSearchRetriever(
        document_store=mock_store,
        stopwords=ENGLISH_STOPWORDS
    )


# ---------------------------------------------------------------------------
# Tests: Keyword Extraction
# ---------------------------------------------------------------------------


class TestKeywordExtraction:
    """Tests for _extract_keywords method."""

    def test_basic_keyword_extraction(self, retriever: WebSearchRetriever) -> None:
        """Extract keywords from a simple query."""
        keywords = retriever._extract_keywords("blood pressure hypertension")
        assert "blood" in keywords
        assert "pressure" in keywords
        assert "hypertension" in keywords

    def test_stopword_removal(self, retriever: WebSearchRetriever) -> None:
        """Remove stopwords from query."""
        keywords = retriever._extract_keywords("the and or blood pressure")
        assert "blood" in keywords
        assert "pressure" in keywords
        assert "the" not in keywords
        assert "and" not in keywords

    def test_lowercase_normalization(self, retriever: WebSearchRetriever) -> None:
        """Normalize keywords to lowercase."""
        keywords = retriever._extract_keywords("HYPERTENSION Blood Pressure")
        assert all(k.islower() for k in keywords)

    def test_empty_query_returns_empty_list(self, retriever: WebSearchRetriever) -> None:
        """Empty query should return empty keyword list."""
        keywords = retriever._extract_keywords("")
        assert keywords == []

    def test_only_stopwords_returns_empty_list(self, retriever: WebSearchRetriever) -> None:
        """Query with only stopwords returns empty list."""
        keywords = retriever._extract_keywords("the and or")
        assert keywords == []


# ---------------------------------------------------------------------------
# Tests: Document Scoring
# ---------------------------------------------------------------------------


class TestDocumentScoring:
    """Tests for _compute_score method."""

    def test_score_with_keyword_match(self, retriever: WebSearchRetriever) -> None:
        """Document with matching keywords should have positive score."""
        doc = Document(
            doc_id="test",
            text="This document discusses hypertension and blood pressure management.",
            url="https://example.com/test",
        )
        score = retriever._compute_score(doc, ["hypertension"])
        assert score > 0

    def test_score_with_no_keyword_match(self, retriever: WebSearchRetriever) -> None:
        """Document without matching keywords should score 0."""
        doc = Document(
            doc_id="test",
            text="This is about cars and motorcycles.",
            url="https://example.com/test",
        )
        score = retriever._compute_score(doc, ["hypertension"])
        assert score == 0

    def test_score_boosts_title_matches(self, retriever: WebSearchRetriever) -> None:
        """Score should be higher when keywords appear in title."""
        doc_without_title = Document(
            doc_id="test1",
            text="This document discusses hypertension extensively in the body.",
            url="https://example.com/test",
        )
        doc_with_title = Document(
            doc_id="test2",
            text="This document discusses hypertension extensively in the body.",
            url="https://example.com/test",
            metadata={"title": "Understanding Hypertension"},
        )
        score_without = retriever._compute_score(doc_without_title, ["hypertension"])
        score_with = retriever._compute_score(doc_with_title, ["hypertension"])
        assert score_with > score_without

    def test_empty_document_returns_zero_score(self, retriever: WebSearchRetriever) -> None:
        """Empty document should score 0."""
        doc = Document(doc_id="test", text="", url="https://example.com/test")
        score = retriever._compute_score(doc, ["hypertension"])
        assert score == 0

    def test_empty_keyword_list_returns_zero_score(self, retriever: WebSearchRetriever) -> None:
        """Empty keyword list should score 0."""
        doc = Document(
            doc_id="test",
            text="This document has content.",
            url="https://example.com/test",
        )
        score = retriever._compute_score(doc, [])
        assert score == 0


# ---------------------------------------------------------------------------
# Tests: Full Retrieval
# ---------------------------------------------------------------------------


class TestFullRetrieval:
    """Tests for the retrieve method."""

    def test_retrieve_returns_matching_documents(
        self, retriever: WebSearchRetriever
    ) -> None:
        """Retrieve should return documents matching query keywords."""
        query = Query(text="hypertension blood pressure")
        results = retriever.retrieve(query, top_k=10)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedDocument) for r in results)

    def test_retrieve_returns_top_k_results(
        self, retriever: WebSearchRetriever
    ) -> None:
        """Retrieve should return at most top_k results."""
        query = Query(text="treatment pain medication")
        results = retriever.retrieve(query, top_k=2)
        assert len(results) <= 2

    def test_retrieve_ranks_by_relevance(
        self, retriever: WebSearchRetriever, sample_documents: dict[str, Document]
    ) -> None:
        """Results should be ranked by relevance score (descending)."""
        query = Query(text="hypertension blood pressure")
        results = retriever.retrieve(query, top_k=10)
        # Check that scores are in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_retrieve_with_empty_query_returns_empty(
        self, retriever: WebSearchRetriever
    ) -> None:
        """Retrieve with empty query should return empty list."""
        query = Query(text="")
        results = retriever.retrieve(query, top_k=10)
        assert results == []

    def test_retrieve_with_only_stopwords_returns_empty(
        self, retriever: WebSearchRetriever
    ) -> None:
        """Retrieve with only stopwords should return empty list."""
        query = Query(text="the and or")
        results = retriever.retrieve(query, top_k=10)
        assert results == []

    def test_retrieve_with_no_matching_documents_returns_empty(
        self, retriever: WebSearchRetriever
    ) -> None:
        """Retrieve with no matching documents should return empty list."""
        query = Query(text="xyz nonexistent medical term")
        results = retriever.retrieve(query, top_k=10)
        assert results == []

    def test_retrieve_score_is_normalized(
        self, retriever: WebSearchRetriever
    ) -> None:
        """Retrieved document scores should be in range [0.0, 1.0]."""
        query = Query(text="treatment medication therapy")
        results = retriever.retrieve(query, top_k=10)
        for result in results:
            assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Tests: Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests with real document interactions."""

    def test_retriever_with_empty_store(self) -> None:
        """Retriever should handle empty document store gracefully."""
        empty_store = MockDocumentStore({})
        retriever = WebSearchRetriever(document_store=empty_store)
        query = Query(text="hypertension")
        results = retriever.retrieve(query, top_k=10)
        assert results == []

    def test_retriever_with_missing_document(self, mock_store: MockDocumentStore) -> None:
        """Retriever should skip documents that cannot be retrieved."""
        # Add a fake ID that doesn't exist
        mock_store._store["fake_doc"] = Document(
            doc_id="real_doc",
            text="hypertension content here",
            url="https://example.com",
        )
        retriever = WebSearchRetriever(document_store=mock_store)
        query = Query(text="hypertension")
        # Should not crash, just skip missing docs
        results = retriever.retrieve(query, top_k=10)
        assert len(results) > 0

    def test_retriever_without_list_all_ids_support(self) -> None:
        """Retriever should handle document stores that don't support list_all_ids."""
        class LimitedStore(DocumentStore):
            """Store that doesn't implement list_all_ids."""
            def add_documents(self, documents: list[Document]) -> None:
                pass
            def get_by_ids(self, doc_ids: list[str]) -> list[Document]:
                return []
            def get_by_id(self, doc_id: str) -> Document | None:
                return None

        limited_store = LimitedStore()
        retriever = WebSearchRetriever(document_store=limited_store)
        query = Query(text="hypertension")
        results = retriever.retrieve(query, top_k=10)
        assert results == []
