"""LSI retriever service — orchestrates TF-IDF, LSI, vector DB, and document store.

This is the main entry-point for retrieval. It implements a two-tier storage architecture:
    - Vector DB (ChromaDB): Stores doc IDs + embeddings for fast similarity search
    - Document Store: Stores full document text and metadata

Retrieval flow:
    IndexedCorpus (docs) → TfidfProcessor.fit() → LSIModel.fit() → store vectors
    IndexedCorpus (query) → TfidfProcessor.transform() → LSIModel.project() → search
"""

from __future__ import annotations

from core.interfaces import BaseRetriever, BaseRepository, DocumentStore, IndexedCorpus
from core.models import Query, RetrievedDocument

from .lsi_model import LSIModel
from .tfidf_processor import TfidfProcessor


class LSIRetriever(BaseRetriever):
    """Retriever based on TF-IDF + TruncatedSVD (LSI) with two-tier storage.

    Orchestrates:
        1. TfidfProcessor    — builds/queries the TF-IDF matrix.
        2. LSIModel          — reduces dimensionality via SVD.
        3. BaseRepository    — stores/searches document vectors (ChromaDB).
        4. DocumentStore     — stores full document text and metadata.

    Storage architecture:
        - Vector DB: ID + embedding + URL (for fast similarity search)
        - Document Store: ID + full text + metadata (for retrieval)
    """

    # Default similarity threshold - results below this score are filtered out
    DEFAULT_SIMILARITY_THRESHOLD = 0.25

    def __init__(
        self,
        repository: BaseRepository,
        document_store: DocumentStore,
        model_dir: str = "models/lsi",
        n_components: int = 100,
        similarity_threshold: float | None = None,
        feedback_service: "RelevanceFeedbackService | None" = None,
    ) -> None:
        """Initialize with repository, document store, and hyper-parameters.

        Args:
            repository: Vector storage backend (e.g. ChromaRepository).
            document_store: Full text storage backend (e.g. FileSystemDocumentStore).
            model_dir: Directory for persisting model artifacts.
            n_components: Number of latent LSI dimensions.
            similarity_threshold: Minimum similarity score (0-1) for results.
                Results below this threshold are filtered out. If None, uses
                DEFAULT_SIMILARITY_THRESHOLD (0.25).
            feedback_service: Optional
                :class:`~plugins.feedback.RelevanceFeedbackService`. When
                provided, the latent query vector is re-weighted via Rocchio
                using any judgments the user has previously recorded for
                this query text. When None, retrieval is unchanged.
        """
        self.repository = repository
        self.document_store = document_store
        self.model_dir = model_dir
        self.n_components = n_components
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.DEFAULT_SIMILARITY_THRESHOLD
        )
        self.feedback_service = feedback_service

        # Sub-components — initialized during fit or load
        self.tfidf: TfidfProcessor | None = None
        self.model: LSIModel | None = None

        # Dynamic-indexing counters (Conf_2): how many documents were folded in
        # since the last full fit. Used by needs_rebalance() to flag when a
        # refit ("balanceo") is warranted to recapture new vocabulary.
        self._base_doc_count = 0
        self._incremental_doc_count = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, corpus: IndexedCorpus) -> None:
        """Fit the full retrieval pipeline from an indexed corpus.

        Steps:
            1. Build TF-IDF matrix from the IndexedCorpus (uses inverted_index).
            2. Fit LSI (SVD) on the TF-IDF matrix → document vectors.
            3. Store documents in document store (full text + metadata).
            4. Store IDs + vectors + URLs in vector repository.

        Args:
            corpus: Preprocessed corpus from the indexer, containing
                documents, processed texts, inverted index, and vocabulary.
        """
        # 1. TF-IDF - build matrix from inverted index
        self.tfidf = TfidfProcessor()
        tfidf_matrix = self.tfidf.fit(corpus)

        # 2. LSI
        self.model = LSIModel(n_components=self.n_components)
        embeddings = self.model.fit(tfidf_matrix)

        # 3. Store full documents in document store
        self.document_store.add_documents(corpus.documents)

        # 4. Store vectors in vector repository (IDs + embeddings + URLs)
        self.repository.add_documents(corpus.documents, embeddings=embeddings)

        # A full fit resets the dynamic-indexing counters: every document is now
        # part of the base index built from the current vocabulary.
        self._base_doc_count = len(corpus.documents)
        self._incremental_doc_count = 0

    # ------------------------------------------------------------------
    # Dynamic indexing (Conf_2 — actualización dinámica)
    # ------------------------------------------------------------------

    def add_documents(self, corpus: IndexedCorpus) -> int:
        """Incrementally index new documents via LSI folding-in.

        Implements the "Incremental" path of dynamic indexing from the lecture
        (Conf_2): new documents are added *without* re-fitting the SVD. Each
        document is weighted with the existing IDF, projected onto the existing
        latent basis, and appended to the vector repository and document store.

        Terms not present in the fitted vocabulary are ignored (folding-in
        limitation). When too many documents have been folded in, call
        :meth:`needs_rebalance` and, if it returns True, rebuild with
        :meth:`fit` to recapture the new vocabulary ("Balanceo").

        Args:
            corpus: IndexedCorpus of the *new* documents only, built by the
                indexer with the same TextProcessor used for the base corpus.

        Returns:
            Number of documents added.

        Raises:
            RuntimeError: If the retriever has not been fitted or loaded.
        """
        if self.tfidf is None or self.model is None:
            raise RuntimeError(
                "Retriever must be fitted or loaded before add_documents()."
            )

        if not corpus.documents:
            return 0

        # Folding-in: TF-IDF with fixed vocabulary/IDF, then project onto the
        # existing latent space (no SVD re-fit).
        tfidf_matrix = self.tfidf.transform_corpus(corpus)
        embeddings = self.model.project_documents(tfidf_matrix)

        # Append to both stores (ChromaDB.add is incremental — existing vectors
        # are untouched).
        self.document_store.add_documents(corpus.documents)
        self.repository.add_documents(corpus.documents, embeddings=embeddings)

        self._incremental_doc_count += len(corpus.documents)
        return len(corpus.documents)

    @property
    def incremental_fraction(self) -> float:
        """Fraction of the index added by folding-in since the last full fit."""
        total = self._base_doc_count + self._incremental_doc_count
        if total == 0:
            return 0.0
        return self._incremental_doc_count / total

    def needs_rebalance(self, threshold: float = 0.2) -> bool:
        """Whether a full refit ("balanceo") is warranted.

        Returns True once the share of folded-in documents exceeds *threshold*.
        Past that point the fixed vocabulary increasingly fails to represent the
        newer documents, so rebuilding with :meth:`fit` restores retrieval
        quality.

        Args:
            threshold: Incremental-fraction cut-off in [0, 1]. Default 0.2 (20%).

        Returns:
            True if ``incremental_fraction > threshold``.
        """
        return self.incremental_fraction > threshold

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: Query,
        top_k: int = 10,
        threshold: float | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve the top-k most relevant documents for a query.

        Two-phase retrieval:
            1. Vector similarity search → returns doc IDs and scores
            2. Fetch full documents from document store by IDs

        Results are filtered by similarity threshold and returned in ranked order
        (highest score first).

        The query must contain an IndexedCorpus (built by the pipeline).
        The TF-IDF processor filters terms not in its vocabulary, then
        projects through LSI for vector search.

        Args:
            query: User query with indexed_corpus populated.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0-1) for results. If None,
                uses the instance's similarity_threshold. Results below this
                threshold are filtered out.

        Returns:
            Ranked list of retrieved documents with full text, filtered by
            similarity threshold and sorted by score (descending).

        Raises:
            RuntimeError: If the retriever has not been fitted or loaded.
            ValueError: If query.indexed_corpus is None.
        """
        if self.tfidf is None or self.model is None:
            raise RuntimeError(
                "Retriever must be fitted or loaded before use."
            )

        if query.indexed_corpus is None:
            raise ValueError(
                "Query must have indexed_corpus populated. "
                "Build it with the indexer before calling retrieve()."
            )

        # Use provided threshold or fall back to instance default
        min_score = threshold if threshold is not None else self.similarity_threshold

        # Phase 1: Vector similarity search
        query_tfidf = self.tfidf.transform(query.indexed_corpus)
        query_vector = self.model.project_query(query_tfidf)

        # Apply explicit relevance feedback (Rocchio) when a feedback service
        # is wired and the user has previously judged docs for this query.
        # No-op when feedback_service is None or no judgments exist.
        if self.feedback_service is not None:
            try:
                query_vector = self.feedback_service.apply_to_query(
                    query.text,
                    query_vector,
                    self.repository.get_embedding,
                )
            except NotImplementedError:
                # Repository does not expose embedding lookup — silently
                # skip feedback rather than crash retrieval.
                pass

        # Get ranked (doc_id, score) pairs from vector DB
        results = self.repository.search_similar(query_vector, top_k=top_k)

        if not results:
            return []

        # Filter by similarity threshold
        filtered_results = [
            (doc_id, score) for doc_id, score in results if score >= min_score
        ]

        if not filtered_results:
            return []

        # Phase 2: Fetch full documents from document store
        doc_ids = [doc_id for doc_id, _ in filtered_results]
        score_map = {doc_id: score for doc_id, score in filtered_results}

        documents = self.document_store.get_by_ids(doc_ids)
        doc_map = {doc.doc_id: doc for doc in documents}

        # Build result list preserving original ranking order from vector search
        retrieved = []
        for doc_id in doc_ids:
            if doc_id in doc_map:
                retrieved.append(
                    RetrievedDocument(
                        document=doc_map[doc_id],
                        score=score_map[doc_id],
                    )
                )

        return retrieved

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str | None = None) -> None:
        """Persist TF-IDF and LSI model artifacts to disk.

        Args:
            model_dir: Override for the default model directory.
        """
        path = model_dir or self.model_dir
        if self.tfidf:
            self.tfidf.save(path)
        if self.model:
            self.model.save(path)

    @classmethod
    def load(
        cls,
        repository: BaseRepository,
        document_store: DocumentStore,
        model_dir: str = "models/lsi",
        similarity_threshold: float | None = None,
    ) -> "LSIRetriever":
        """Restore a fitted retriever from persisted artifacts.

        Loads TF-IDF vectorizer and SVD model from disk.

        Args:
            repository: Vector storage backend.
            document_store: Full text storage backend.
            model_dir: Directory containing saved artifacts.
            similarity_threshold: Minimum similarity score for results.
                If None, uses DEFAULT_SIMILARITY_THRESHOLD.

        Returns:
            Ready-to-use ``LSIRetriever`` instance.
        """
        instance = cls(
            repository=repository,
            document_store=document_store,
            model_dir=model_dir,
            similarity_threshold=similarity_threshold,
        )
        instance.tfidf = TfidfProcessor.load(model_dir)
        instance.model = LSIModel.load(model_dir)
        # Treat the persisted corpus as the base index so the dynamic-indexing
        # counters start from the right denominator.
        instance._base_doc_count = instance.tfidf.n_docs
        instance._incremental_doc_count = 0
        return instance

