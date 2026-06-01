"""Abstract interfaces used by mandatory SRI modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.models import Query, Document, RetrievedDocument, RAGResponse, PipelineContext


# ---------------------------------------------------------------------------
# Adapter: data container for preprocessed corpus from the indexer
# ---------------------------------------------------------------------------


@dataclass
class IndexedCorpus:
    """Data container that bridges the indexer and TF-IDF processor.

    The indexer produces an IndexedCorpus; the retriever consumes it.
    This keeps the two decoupled while ensuring TF-IDF gets exactly
    what it needs.

    Attributes:
        documents: Original Document objects to store in the repository.
        processed_texts: Preprocessed text for each document (same order).
            The indexer applies tokenization, lemmatization, stopword removal, etc.
        inverted_index: Mapping of term → list of (doc_index, term_frequency).
            Used to construct the TF-IDF matrix efficiently.
            Example: {'hipertensión': [(0, 2), (3, 1)], 'diabetes': [(1, 3), (2, 1)]}
        vocabulary: Ordered list of all unique terms. Maps term → term_index.
            This defines the column order of the TF-IDF matrix.
    """

    documents: list[Document]
    processed_texts: list[str]
    inverted_index: dict[str, list[tuple[int, int]]]
    vocabulary: list[str]
    corrected_text: str | None = None

    def __post_init__(self) -> None:
        if len(self.documents) != len(self.processed_texts):
            raise ValueError(
                f"documents ({len(self.documents)}) and processed_texts "
                f"({len(self.processed_texts)}) must have the same length"
            )

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def doc_ids(self) -> list[str]:
        """Document IDs extracted from the documents list."""
        return [doc.doc_id for doc in self.documents]


# ---------------------------------------------------------------------------
# Data-layer contracts
# ---------------------------------------------------------------------------


class DocumentStore(ABC):
    """Protocol for full document content storage, decoupled from vector search.

    This interface separates document content storage from vector embeddings.
    The vector database stores only IDs and embeddings, while this store
    maintains the full document text and metadata.
    """

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Store full document content and metadata.

        Args:
            documents: List of documents to persist.
        """
        raise NotImplementedError

    @abstractmethod
    def get_by_ids(self, doc_ids: list[str]) -> list[Document]:
        """Retrieve multiple documents by their IDs.

        Args:
            doc_ids: List of document IDs to fetch.

        Returns:
            List of documents in the same order as doc_ids (skips missing IDs).
        """
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a single document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            Document if found, None otherwise.
        """
        raise NotImplementedError

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists without loading it.

        Default implementation uses get_by_id. Subclasses may override
        for more efficient existence checks.

        Args:
            doc_id: Document identifier.

        Returns:
            True if the document exists, False otherwise.
        """
        return self.get_by_id(doc_id) is not None

    def delete(self, doc_id: str) -> bool:
        """Delete a document from storage.

        Args:
            doc_id: Document identifier.

        Returns:
            True if the document was deleted, False if it didn't exist.

        Note:
            Default implementation raises NotImplementedError.
            Subclasses should override if deletion is supported.
        """
        raise NotImplementedError("Delete not supported by this store")

    def list_all_ids(self) -> list[str]:
        """List all document IDs in the store.

        Used by retrieval strategies (like web_search) to iterate over available documents
        for fallback searches when primary retrieval strategies don't return enough results.

        Default implementation raises NotImplementedError.
        Subclasses should override if iteration is supported.

        Returns:
            List of all document IDs in the store.

        Note:
            For large collections, implement this efficiently (e.g., streaming from disk)
            to avoid loading all documents into memory at once.
        """
        raise NotImplementedError("List all IDs not supported by this store")


class BaseRepository(ABC):
    """Protocol for vector storage and similarity search.

    This repository stores document IDs, embeddings (vectors), and minimal
    metadata (e.g. URL). Full document text is stored separately in a DocumentStore.
    """

    @abstractmethod
    def add_documents(self, documents: list[Document], embeddings: list[list[float]] | None = None) -> None:
        """Store document IDs, vectors, and minimal metadata.

        Args:
            documents: Documents to extract IDs and URLs from.
            embeddings: Pre-calculated latent vectors for LSI search.
        """
        raise NotImplementedError

    def get_embedding(self, doc_id: str) -> list[float] | None:
        """Return the stored embedding for ``doc_id``, or None if unknown.

        Used by feedback algorithms (Rocchio) that need the latent vector
        of specific documents to nudge the query vector toward / away from
        them. Default implementation raises NotImplementedError; subclasses
        should override when they have efficient embedding lookup.

        Args:
            doc_id: Document identifier.

        Returns:
            Latent vector if the doc is known, ``None`` otherwise.
        """
        raise NotImplementedError("Embedding lookup not supported by this repository")

    @abstractmethod
    def search_similar(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Search for documents similar to the given vector.

        Args:
            query_vector: Latent vector representing the query.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, score) tuples ranked by relevance.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Retrieval strategy contract
# ---------------------------------------------------------------------------


class BaseRetriever(ABC):
    """Strategy interface for document retrieval."""

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10) -> list[RetrievedDocument]:
        """Execute a retrieval strategy for the given query."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Re-ranking strategy contract
# ---------------------------------------------------------------------------


class BaseRanker(ABC):
    """Strategy interface for re-ranking retrieved documents before generation.

    A re-ranker sits between the retriever and the RAG generator. It receives
    the top-k documents returned by the retriever and reorders them so the
    most relevant ones appear first in the LLM context window.

    Implementations may use any signal: BM25 overlap, cross-encoder scores,
    or hybrid combinations of lexical and semantic similarity.
    """

    @abstractmethod
    def rerank(
        self,
        query: Query,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Re-rank retrieved documents by relevance to the query.

        Args:
            query: The original user query (provides raw text for scoring).
            documents: Documents returned by the retriever, in retriever order.

        Returns:
            The same documents reordered by descending combined relevance score.
            Scores on each RetrievedDocument may be updated to reflect the new
            ranking signal.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RAG generation strategy contract
# ---------------------------------------------------------------------------


class BaseRAG(ABC):
    """Strategy interface for profile-aware answer generation.

    A concrete implementation receives the original query, the retrieved
    documents as context, and produces a RAGResponse. The profile is
    read from query.user_profile. If user_profile is None, the
    implementation defaults to PATIENT profile.
    """

    @abstractmethod
    def generate(
        self,
        query: Query,
        retrieved_docs: list[RetrievedDocument],
        max_context_docs: int = 3,
    ) -> RAGResponse:
        """Generate a profile-adapted answer.

        Args:
            query: The original user query, may carry a UserProfile.
            retrieved_docs: Ranked documents from the retriever.
            max_context_docs: Maximum number of documents to include in
                the LLM context window. Default is 3 to stay within limits.

        Returns:
            RAGResponse with the generated answer and provenance metadata.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Optional plugin contract (microkernel extension points)
# ---------------------------------------------------------------------------


class Plugin(ABC):
    """Contract for optional pipeline plugins (microkernel architecture).

    Plugins extend the pipeline at well-known hook points without the core
    importing them directly. Each plugin declares the hook it attaches to and
    transforms the shared :class:`~core.models.PipelineContext` in place (or
    returns a new one). If a plugin is never registered, the pipeline behaves
    exactly as if it did not exist.

    Recognised hooks:
        - ``pre_retrieval``  — runs before the retriever (e.g. query expansion).
        - ``post_retrieval`` — runs after retrieval, before ranking.
        - ``post_ranking``   — runs after ranking, before answer generation.
    """

    @abstractmethod
    def hook_name(self) -> str:
        """Return the hook this plugin attaches to.

        Returns:
            One of ``"pre_retrieval"``, ``"post_retrieval"``, ``"post_ranking"``.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Transform the pipeline context at this plugin's hook point.

        Args:
            context: The shared pipeline context. Plugins read/modify
                ``context.query``, ``context.results``, and record what they
                did under ``context.metadata``.

        Returns:
            The (possibly mutated) context for the next stage.
        """
        raise NotImplementedError
