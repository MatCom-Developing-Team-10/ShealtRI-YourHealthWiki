"""Repository implementation using ChromaDB for vector storage and similarity search.

This repository stores only document IDs, embeddings, and minimal metadata (URLs).
Full document text is stored separately in a DocumentStore implementation.
"""

import chromadb

from core.interfaces import BaseRepository
from core.models import Document


class ChromaRepository(BaseRepository):
    """Vector storage using ChromaDB for efficient similarity search.

    Stores only:
        - Document IDs
        - Latent vectors (embeddings from LSI)
        - Minimal metadata (URL only)

    Full document text is stored separately in a DocumentStore.
    This separation improves scalability and reduces vector DB costs.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "medical_documents",
    ) -> None:
        """Initialize ChromaDB client and collection.

        Args:
            persist_directory: Path where ChromaDB will store its data.
            collection_name: Name of the collection in ChromaDB.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document], embeddings: list[list[float]] | None = None) -> None:
        """Store document IDs, vectors, and minimal metadata (URLs only).

        Full document text is NOT stored here. Use a DocumentStore for that.

        Args:
            documents: Documents to extract IDs and URLs from.
            embeddings: Pre-calculated latent vectors for LSI search.
        """
        ids = [doc.doc_id for doc in documents]
        metadatas = [{"url": doc.url} for doc in documents]

        self.collection.add(
            ids=ids,
            documents=None,  # Don't store full text - use DocumentStore instead
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search_similar(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Search for documents similar to the given vector.

        Args:
            query_vector: Latent vector representing the query.
            top_k: Number of documents to return.

        Returns:
            List of (doc_id, score) tuples ranked by relevance.
            Score is computed as 1 - distance (higher is better).
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        ranked_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                # Convert distance to similarity score (1 - distance)
                # ChromaDB uses cosine distance by default
                score = 1.0 - results["distances"][0][i]
                ranked_results.append((doc_id, score))

        return ranked_results

