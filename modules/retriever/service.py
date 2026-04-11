"""LSI retriever service — orchestrates TF-IDF, LSI, vector DB, and document store.

This is the main entry-point for retrieval. It implements a two-tier storage architecture:
    - Vector DB (ChromaDB): Stores doc IDs + embeddings for fast similarity search
    - Document Store: Stores full document text and metadata

Retrieval pipeline:
    Query (raw text) → TextProcessor.process(is_query=True)
                    → TfidfProcessor.transform()
                    → LSIModel.project_query()
                    → ChromaRepository.search_similar()
                    → DocumentStore.get_by_ids()
                    → RetrievedDocument list (ranked)
"""

from __future__ import annotations

from core.interfaces import BaseRetriever, BaseRepository, DocumentStore, IndexedCorpus
from core.models import Query, RetrievedDocument
from modules.text_processor import TextProcessor

from .lsi_model import LSIModel
from .tfidf_processor import TfidfProcessor


class LSIRetriever(BaseRetriever):
    """Retriever based on TF-IDF + TruncatedSVD (LSI) with two-tier storage.

    Orchestrates:
        1. TextProcessor     — normalizes and preprocesses query text.
        2. TfidfProcessor    — builds/queries the TF-IDF matrix.
        3. LSIModel          — reduces dimensionality via SVD.
        4. BaseRepository    — stores/searches document vectors (ChromaDB).
        5. DocumentStore     — stores full document text and metadata.

    Storage architecture:
        - Vector DB: ID + embedding + URL (for fast similarity search)
        - Document Store: ID + full text + metadata (for retrieval)
    """

    # Default similarity threshold - results below this score are filtered out
    DEFAULT_SIMILARITY_THRESHOLD = 0.4

    def __init__(
        self,
        repository: BaseRepository,
        document_store: DocumentStore,
        text_processor: TextProcessor | None = None,
        model_dir: str = "models/lsi",
        n_components: int = 100,
        tfidf_max_features: int | None = 10_000,
        tfidf_min_df: int = 1,
        tfidf_max_df: float = 0.95,
        similarity_threshold: float | None = None,
    ) -> None:
        """Initialize with repository, document store, and hyper-parameters.

        Args:
            repository: Vector storage backend (e.g. ChromaRepository).
            document_store: Full text storage backend (e.g. FileSystemDocumentStore).
            text_processor: TextProcessor for query preprocessing. If None, must be
                set via set_text_processor() before calling retrieve().
            model_dir: Directory for persisting model artifacts.
            n_components: Number of latent LSI dimensions.
            tfidf_max_features: Cap on TF-IDF vocabulary size (ignored when
                the corpus supplies its own vocabulary).
            tfidf_min_df: Minimum document frequency for TF-IDF terms.
            tfidf_max_df: Maximum document frequency for TF-IDF terms.
            similarity_threshold: Minimum similarity score (0-1) for results.
                Results below this threshold are filtered out. If None, uses
                DEFAULT_SIMILARITY_THRESHOLD (0.4).
        """
        self.repository = repository
        self.document_store = document_store
        self.text_processor = text_processor
        self.model_dir = model_dir
        self.n_components = n_components
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.DEFAULT_SIMILARITY_THRESHOLD
        )

        # Sub-components — initialized during fit or load
        self.tfidf: TfidfProcessor | None = None
        self.model: LSIModel | None = None

        # Store TF-IDF hyperparameters for deferred construction
        self._tfidf_max_features = tfidf_max_features
        self._tfidf_min_df = tfidf_min_df
        self._tfidf_max_df = tfidf_max_df

    def set_text_processor(self, text_processor: TextProcessor) -> None:
        """Set or override the TextProcessor for query preprocessing.

        Args:
            text_processor: TextProcessor instance for query preprocessing.
        """
        self.text_processor = text_processor

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, corpus: IndexedCorpus) -> None:
        """Fit the full retrieval pipeline from an indexed corpus.

        Steps:
            1. Build TF-IDF matrix from the IndexedCorpus.
            2. Fit LSI (SVD) on the TF-IDF matrix → document vectors.
            3. Store documents in document store (full text + metadata).
            4. Store IDs + vectors + URLs in vector repository.

        Args:
            corpus: Preprocessed corpus from the indexer, containing
                documents, processed texts, inverted index, and vocabulary.
        """
        # 1. TF-IDF - build matrix from indexed corpus
        self.tfidf = TfidfProcessor(
            max_features=self._tfidf_max_features,
            min_df=self._tfidf_min_df,
            max_df=self._tfidf_max_df,
        )
        tfidf_matrix = self.tfidf.fit(corpus)

        # 2. LSI
        self.model = LSIModel(n_components=self.n_components)
        embeddings = self.model.fit(tfidf_matrix)

        # 3. Store full documents in document store
        self.document_store.add_documents(corpus.documents)

        # 4. Store vectors in vector repository (IDs + embeddings + URLs)
        self.repository.add_documents(corpus.documents, embeddings=embeddings)

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

        Pipeline:
            1. Preprocess query using TextProcessor (normalize, tokenize, lemmatize)
            2. Transform to TF-IDF vector
            3. Project to LSI latent space
            4. Search ChromaDB for similar documents
            5. Fetch full documents from DocumentStore
            6. Filter by similarity threshold and return ranked results

        Args:
            query: User query with text string.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0-1) for results. If None,
                uses the instance's similarity_threshold. Results below this
                threshold are filtered out.

        Returns:
            Ranked list of retrieved documents with full text, filtered by
            similarity threshold and sorted by score (descending).

        Raises:
            RuntimeError: If the retriever has not been fitted or loaded.
            RuntimeError: If TextProcessor is not set and needed.
        """
        if self.tfidf is None or self.model is None:
            raise RuntimeError(
                "Retriever must be fitted or loaded before use."
            )

        # Use provided threshold or fall back to instance default
        min_score = threshold if threshold is not None else self.similarity_threshold

        # Phase 1: Preprocess query text
        if self.text_processor is not None:
            # Full preprocessing: normalize, tokenize, lemmatize, spell check
            processed_query = self.text_processor.process(query.text, is_query=True)
        else:
            # Fallback: minimal preprocessing (assumes query is already somewhat clean)
            processed_query = query.text.lower().strip()

        # Phase 2: Vector similarity search
        query_tfidf = self.tfidf.transform(processed_query)
        query_vector = self.model.project_query(query_tfidf)

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

        # Phase 3: Fetch full documents from document store
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
        text_processor: TextProcessor | None = None,
        model_dir: str = "models/lsi",
        similarity_threshold: float | None = None,
    ) -> "LSIRetriever":
        """Restore a fitted retriever from persisted artifacts.

        Loads TF-IDF vectorizer and SVD model from disk.

        Args:
            repository: Vector storage backend.
            document_store: Full text storage backend.
            text_processor: TextProcessor for query preprocessing.
            model_dir: Directory containing saved artifacts.
            similarity_threshold: Minimum similarity score for results.
                If None, uses DEFAULT_SIMILARITY_THRESHOLD.

        Returns:
            Ready-to-use LSIRetriever instance.
        """
        instance = cls(
            repository=repository,
            document_store=document_store,
            text_processor=text_processor,
            model_dir=model_dir,
            similarity_threshold=similarity_threshold,
        )
        instance.tfidf = TfidfProcessor.load(model_dir)
        instance.model = LSIModel.load(model_dir)
        return instance
