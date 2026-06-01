"""Repository implementation using ChromaDB for vector storage and similarity search.

This repository stores only document IDs, embeddings, and minimal metadata (URLs).
Full document text is stored separately in a DocumentStore implementation.

Collection naming convention:
    CORPUS_COLLECTION     — curated documents indexed from the crawler.
    WEB_SEARCH_COLLECTION — documents fetched from external web searches (future).
    Never mix both sources in the same collection.
"""

import chromadb

from core.interfaces import BaseRepository
from core.models import Document

# ChromaDB collection names — one per data source to prevent corpus contamination.
CORPUS_COLLECTION = "corpus_curated"
WEB_SEARCH_COLLECTION = "web_search_results"

# Fallback cap on records per collection.add() call when the client does not
# report its own limit. ChromaDB rejects single adds larger than ~5461.
_DEFAULT_MAX_BATCH = 5000


class ChromaRepository(BaseRepository):
    """Vector storage using ChromaDB for efficient similarity search.

    Stores only:
        - Document IDs
        - Latent vectors (embeddings from LSI)
        - Minimal metadata (URL only)

    Full document text is stored separately in a DocumentStore.
    This separation improves scalability and reduces vector DB costs.

    Use CORPUS_COLLECTION for the curated crawler corpus and
    WEB_SEARCH_COLLECTION for any externally fetched web documents.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = CORPUS_COLLECTION,
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

        # Guard against a documents/embeddings length mismatch up front. Without
        # this check the batch loop below would slice misaligned embeddings and
        # leave the collection partially populated before ChromaDB finally raises.
        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and documents ({len(ids)}) "
                "must have the same length"
            )

        # ChromaDB rejects a single add() larger than its max batch size, so we
        # insert in chunks. Large corpora (tens of thousands of pages) exceed
        # this limit in one call.
        batch_size = self._max_batch_size()
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=None,  # Don't store full text - use DocumentStore instead
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end] if embeddings is not None else None,
            )

    def _max_batch_size(self) -> int:
        """Return the safe number of records per ``collection.add()`` call.

        Honors the ChromaDB client's reported limit when available, falling
        back to a conservative constant otherwise.
        """
        limit = getattr(self.client, "max_batch_size", None)
        if isinstance(limit, int) and limit > 0:
            return limit
        return _DEFAULT_MAX_BATCH

    def get_embedding(self, doc_id: str) -> list[float] | None:
        """Return the stored embedding for ``doc_id`` or None if unknown.

        Used by relevance-feedback algorithms (Rocchio) that need the latent
        vector of specific documents. ChromaDB's ``include=['embeddings']``
        flag returns vectors alongside the IDs query.
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["embeddings"])
        except Exception as exc:  # pragma: no cover - defensive
            import logging
            logging.getLogger(__name__).warning(
                "ChromaDB embedding fetch failed for %s: %s", doc_id, exc
            )
            return None

        if result is None:
            return None
        vectors = result.get("embeddings")
        # ChromaDB may return ``None``, an empty list, or a numpy array — the
        # last case crashes truthy checks (``if not vectors``) when len > 1.
        # Use explicit length / None tests instead.
        if vectors is None or len(vectors) == 0:
            return None
        first = vectors[0]
        if first is None or len(first) == 0:
            return None
        # Normalise numpy arrays to plain list[float].
        return [float(x) for x in first]

    def search_similar(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Search for documents similar to the given vector.

        Args:
            query_vector: Latent vector representing the query.
            top_k: Number of documents to return.

        Returns:
            List of (doc_id, score) tuples ranked by relevance.
            Score is computed as 1 - distance (higher is better, range 0-1).
            Returns empty list if collection is empty or query fails.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
            )
        except Exception as e:
            # Log error and return empty results rather than crashing
            import logging
            logging.getLogger(__name__).error(f"ChromaDB query failed: {e}")
            return []

        ranked_results = []

        # Validate response structure
        if not results or not results.get("ids") or not results["ids"]:
            return []

        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0]

        # Ensure we have matching lengths
        if len(ids) != len(distances):
            import logging
            logging.getLogger(__name__).warning(
                f"Mismatched results: {len(ids)} ids vs {len(distances)} distances"
            )
            return []

        for i in range(len(ids)):
            doc_id = ids[i]
            distance = distances[i]

            # Handle potential None or invalid distance values
            if distance is None:
                continue

            # Convert distance to similarity score (1 - distance)
            # ChromaDB uses cosine distance (0 = identical, 2 = opposite)
            # Clamp to [0, 1] range for safety
            score = max(0.0, min(1.0, 1.0 - distance))
            ranked_results.append((doc_id, score))

        return ranked_results
