"""LSI retriever service — orchestrates TF-IDF, LSI, vector DB, and document store.

This is the main entry-point for retrieval. It implements a two-tier storage architecture:
    - Vector DB (ChromaDB): Stores doc IDs + embeddings for fast similarity search
    - Document Store: Stores full document text and metadata

Retrieval flow:
    IndexedCorpus → TfidfProcessor → LSIModel → ChromaRepository (vectors)
                                                → DocumentStore (full text)
"""

from __future__ import annotations

import re

from core.interfaces import BaseRetriever, BaseRepository, DocumentStore, IndexedCorpus
from core.models import Query, RetrievedDocument

from .lsi_model import LSIModel
from .spell_checker import TrieSpellChecker
from .tfidf_processor import TfidfProcessor


class LSIRetriever(BaseRetriever):
    """Retriever based on TF-IDF + TruncatedSVD (LSI) with two-tier storage.

    Orchestrates:
        1. TfidfProcessor    — builds/queries the TF-IDF matrix.
        2. LSIModel          — reduces dimensionality via SVD.
        3. BaseRepository    — stores/searches document vectors (ChromaDB).
        4. DocumentStore     — stores full document text and metadata.
        5. TrieSpellChecker  — corrects query typos before vectorization.

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
        model_dir: str = "models/lsi",
        n_components: int = 100,
        max_spell_distance: int = 2,
        tfidf_max_features: int | None = 10_000,
        tfidf_min_df: int = 1,
        tfidf_max_df: float = 0.95,
        similarity_threshold: float | None = None,
    ) -> None:
        """Initialize with repository, document store, and hyper-parameters.

        Args:
            repository: Vector storage backend (e.g. ChromaRepository).
            document_store: Full text storage backend (e.g. FileSystemDocumentStore).
            model_dir: Directory for persisting model artifacts.
            n_components: Number of latent LSI dimensions.
            max_spell_distance: Max edit distance for spelling correction.
            tfidf_max_features: Cap on TF-IDF vocabulary size (ignored when
                the corpus supplies its own vocabulary).
            tfidf_min_df: Minimum document frequency for TF-IDF terms.
            tfidf_max_df: Maximum document frequency for TF-IDF terms.
            similarity_threshold: Minimum similarity score (0-1) for results.
                Results below this threshold are filtered out. If None, uses
                DEFAULT_SIMILARITY_THRESHOLD (0.1).
        """
        self.repository = repository
        self.document_store = document_store
        self.model_dir = model_dir
        self.n_components = n_components
        self.max_spell_distance = max_spell_distance
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.DEFAULT_SIMILARITY_THRESHOLD
        )

        # Sub-components — initialized during fit or load
        self.tfidf: TfidfProcessor | None = None
        self.model: LSIModel | None = None
        self.spell_checker: TrieSpellChecker | None = None

        # Store TF-IDF hyperparameters for deferred construction
        self._tfidf_max_features = tfidf_max_features
        self._tfidf_min_df = tfidf_min_df
        self._tfidf_max_df = tfidf_max_df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, corpus: IndexedCorpus) -> None:
        """Fit the full retrieval pipeline from an indexed corpus.

        Steps:
            1. Build TF-IDF matrix from the ``IndexedCorpus``.
            2. Fit LSI (SVD) on the TF-IDF matrix → document vectors.
            3. Store documents in document store (full text + metadata).
            4. Store IDs + vectors + URLs in vector repository.
            5. Initialize the spell-checker with the TF-IDF vocabulary.

        Args:
            corpus: Preprocessed corpus from the indexer, containing
                documents and their processed texts.
        """
        # 1. TF-IDF
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

        # 5. Spell checker from TF-IDF vocabulary
        vocab = self.tfidf.vocabulary
        self.spell_checker = TrieSpellChecker(
            vocabulary=vocab,
            max_distance=self.max_spell_distance,
        )

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

        The query text is cleaned, spell-corrected, projected through
        TF-IDF → LSI, and used to search the vector repository.

        Args:
            query: User query.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0-1) for results. If None,
                uses the instance's similarity_threshold. Results below this
                threshold are filtered out.

        Returns:
            Ranked list of retrieved documents with full text, filtered by
            similarity threshold and sorted by score (descending).

        Raises:
            RuntimeError: If the retriever has not been fitted or loaded.
        """
        if self.tfidf is None or self.model is None:
            raise RuntimeError(
                "Retriever must be fitted or loaded before use."
            )

        # Use provided threshold or fall back to instance default
        min_score = threshold if threshold is not None else self.similarity_threshold

        # Phase 1: Vector similarity search
        clean_text = self._normalize_query(query.text)
        query_tfidf = self.tfidf.transform(clean_text)
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
        max_spell_distance: int = 2,
        similarity_threshold: float | None = None,
    ) -> "LSIRetriever":
        """Restore a fitted retriever from persisted artifacts.

        Loads TF-IDF vectorizer and SVD model from disk, and rebuilds
        the spell checker from the TF-IDF vocabulary.

        Args:
            repository: Vector storage backend.
            document_store: Full text storage backend.
            model_dir: Directory containing saved artifacts.
            max_spell_distance: Max edit distance for spelling correction.
            similarity_threshold: Minimum similarity score for results.
                If None, uses DEFAULT_SIMILARITY_THRESHOLD.

        Returns:
            Ready-to-use ``LSIRetriever`` instance.
        """
        instance = cls(
            repository=repository,
            document_store=document_store,
            model_dir=model_dir,
            max_spell_distance=max_spell_distance,
            similarity_threshold=similarity_threshold,
        )
        instance.tfidf = TfidfProcessor.load(model_dir)
        instance.model = LSIModel.load(model_dir)

        # Rebuild spell checker from the loaded TF-IDF vocabulary
        vocab = instance.tfidf.vocabulary
        instance.spell_checker = TrieSpellChecker(
            vocabulary=vocab,
            max_distance=max_spell_distance,
        )
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_query(self, text: str) -> str:
        """Apply basic cleaning and spell correction to query text."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        if self.spell_checker:
            words = text.split()
            corrected = [self.spell_checker.correct(w) or w for w in words]
            return " ".join(corrected)
        return text
