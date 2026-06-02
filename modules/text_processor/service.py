"""Text preprocessing pipeline for Spanish medical documents using spaCy.

This module provides text preprocessing for the LSI indexing pipeline:
    Raw text → normalize → tokenize → remove stopwords → lemmatize → filter → [spell-check/add to vocab]

Usage:
    processor = TextProcessor()

    # For documents: tokens are added to spell checker vocabulary
    processed = processor.process("La hipertensión arterial causa cefalea")
    # Returns: "hipertensión arterial causar cefalea"
    # Tokens added to spell_checker if configured

    # For queries: tokens are corrected using known vocabulary
    processed = processor.process("hipertensoin", is_query=True)
    # Returns: "hipertensión" (corrected from vocabulary)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

import nltk
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from spacy.language import Language

from core.stopwords import SPANISH_MEDICAL_STOPWORDS
from .spell_checker import TrieSpellChecker


logger = logging.getLogger(__name__)


@dataclass
class TextProcessorConfig:
    """Configuration for text preprocessing.

    Attributes:
        language: Language for lemmatization and stopwords (default: "spanish").
        spacy_model: spaCy model name to use (default: "es_core_news_md").
        min_token_length: Minimum token length to keep (default: 2).
        max_token_length: Maximum token length to keep (default: 50).
        remove_accents: Whether to normalize accented characters (default: False).
            Keep False for Spanish to preserve meaning (año vs ano).
        lowercase: Whether to convert to lowercase (default: True).
        custom_stopwords: Additional stopwords to remove beyond defaults.
        batch_size: Number of documents fed to spaCy's ``nlp.pipe`` per batch
            when processing a collection (default: 64). Larger batches improve
            throughput on big corpora at the cost of higher peak memory.
        n_process: Number of worker processes used by spaCy's ``nlp.pipe`` when
            indexing a large collection (default: 1 = single process). Values
            greater than 1 parallelize lemmatization across CPU cores, which is
            the dominant cost of a cold start. Only applied on the indexing path
            and only when the batch is at least ``multiprocess_threshold`` texts
            (forking has fixed overhead that hurts small batches and single
            queries). See :meth:`TextProcessor.process_many`.
        multiprocess_threshold: Minimum number of texts in a single
            ``process_many`` call before ``n_process`` > 1 is actually used.
            Below this, processing stays single-process to avoid paying the
            multiprocessing fork/IPC overhead on a workload too small to benefit.
    """

    language: str = "spanish"
    spacy_model: str = "es_core_news_md"
    min_token_length: int = 2
    max_token_length: int = 20
    remove_accents: bool = False
    lowercase: bool = True
    custom_stopwords: set[str] = field(default_factory=set)
    batch_size: int = 64
    n_process: int = 1
    multiprocess_threshold: int = 500


class TextProcessor:
    """Preprocesses Spanish medical text for LSI indexing.

    Pipeline stages:
        1. Normalization (lowercase, unicode normalization, cleaning)
        2. Tokenization (spaCy tokenizer for Spanish)
        3. Stopword removal (NLTK Spanish + medical domain)
        4. Lemmatization (spaCy lemmatizer with POS tagging)
        5. Token filtering (length constraints)

    Example:
        processor = TextProcessor()
        result = processor.process("La hipertensión arterial causa dolores de cabeza")
        # Result: "hipertensión arterial causar dolor cabeza"

        # With custom config
        config = TextProcessorConfig(min_token_length=3)
        processor = TextProcessor(config)
    """

    def __init__(self, config: TextProcessorConfig | None = None) -> None:
        """Initialize the text processor.

        Args:
            config: Processing configuration. Uses defaults if None.

        Raises:
            OSError: If the spaCy model is not installed.
                Install with: python -m spacy download es_core_news_md

        Note:
            The spaCy model is loaded once on initialization and reused for all texts.
        """
        self.config = config or TextProcessorConfig()

        # Load spaCy model
        try:
            self._nlp: Language = spacy.load(self.config.spacy_model)
            # Disable unnecessary pipeline components for performance
            # We only need tokenizer, tagger, and lemmatizer
            self._nlp.disable_pipes(["parser", "ner"])
        except OSError as e:
            logger.error(
                f"spaCy model '{self.config.spacy_model}' not found. "
                f"Install with: python -m spacy download {self.config.spacy_model}"
            )
            raise OSError(
                f"spaCy model '{self.config.spacy_model}' not installed. "
                f"Run: python -m spacy download {self.config.spacy_model}"
            ) from e

        self._stopwords = self._load_stopwords()
        self.spell_checker: TrieSpellChecker = TrieSpellChecker()
        self._last_corrections: dict[str, str] = {}

        logger.debug(
            f"TextProcessor initialized: language={self.config.language}, "
            f"model={self.config.spacy_model}, stopwords={len(self._stopwords)}"
        )

    def _load_stopwords(self) -> set[str]:
        """Load Spanish stopwords plus custom domain stopwords.

        Downloads NLTK stopwords on first use if not present.
        """
        try:
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                logger.info("Downloading NLTK stopwords package")
                nltk.download("stopwords", quiet=True)

            base_stopwords = set(nltk_stopwords.words(self.config.language))
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords: {e}. Using only custom stopwords.")
            base_stopwords = set()

        combined = base_stopwords | SPANISH_MEDICAL_STOPWORDS | self.config.custom_stopwords

        logger.debug(
            f"Loaded stopwords: {len(base_stopwords)} NLTK + "
            f"{len(SPANISH_MEDICAL_STOPWORDS)} medical + "
            f"{len(self.config.custom_stopwords)} custom"
        )

        return combined

    def process(
        self, text: str, is_query: bool = False, apply_correction: bool = True
    ) -> str:
        """Apply the full preprocessing pipeline to a single text.

        Pipeline flow:
            1. Normalize (lowercase, unicode, cleaning)
            2. Tokenize + lemmatize in a single spaCy pass
            3. Remove stopwords (by surface form) and filter by length
            4. If is_query=False: add tokens to spell checker vocabulary
               If is_query=True: correct tokens using known vocabulary
               (unless apply_correction=False, see below)

        For indexing many documents at once, prefer :meth:`process_many`, which
        batches the spaCy pipeline and is substantially faster.

        Args:
            text: Raw input text.
            is_query: If False, tokens are added to spell checker vocabulary.
                      If True, tokens are corrected using the vocabulary.
            apply_correction: Only relevant when ``is_query=True``. When False,
                the query is processed verbatim (lemmatized/filtered as usual)
                but spell correction is skipped, so the user's original wording
                is preserved. Ignored on the indexing path.

        Returns:
            Space-joined preprocessed tokens ready for indexer.
        """
        return self.process_many(
            [text], is_query=is_query, apply_correction=apply_correction
        )[0]

    def process_many(
        self,
        texts: list[str],
        is_query: bool = False,
        apply_correction: bool = True,
    ) -> list[str]:
        """Preprocess a batch of texts using spaCy's batched ``nlp.pipe``.

        This is the throughput-oriented entry point used by the indexer. It
        runs the spaCy pipeline once per batch (rather than once per call) and,
        for the indexing path, inserts the deduplicated vocabulary into the
        spell checker in a single pass at the end.

        Each output aligns positionally with its input: ``texts[i]`` produces
        ``result[i]``. Empty or whitespace-only inputs yield ``""``.

        Args:
            texts: Raw input texts.
            is_query: If False, surviving tokens populate the spell checker
                vocabulary. If True, tokens are corrected against it.
            apply_correction: Only relevant when ``is_query=True``. When False,
                spell correction is skipped so the query keeps the user's
                original wording. Ignored on the indexing path.

        Returns:
            One space-joined token string per input text, in the same order.
        """
        if not texts:
            return []

        normalized = [self.normalize(text) if text else "" for text in texts]

        # Parallelize lemmatization across CPU cores on the indexing path only.
        # Queries (is_query=True) and small batches stay single-process: spaCy's
        # multiprocessing forks workers and serializes Docs over a pipe, overhead
        # that only pays off on a large indexing collection.
        effective_n_process = (
            self.config.n_process
            if (
                not is_query
                and self.config.n_process > 1
                and len(normalized) >= self.config.multiprocess_threshold
            )
            else 1
        )

        results: list[str] = []
        vocabulary_tokens: set[str] = set()

        for doc in self._nlp.pipe(
            normalized,
            batch_size=self.config.batch_size,
            n_process=effective_n_process,
        ):
            tokens = self._extract_tokens(doc)
            if is_query:
                if apply_correction:
                    tokens = self._correct_spelling(tokens)
                else:
                    # User opted out of correction: keep tokens verbatim and
                    # clear any corrections recorded by a previous call so
                    # get_last_corrections() reflects this run honestly.
                    self._last_corrections = {}
            else:
                vocabulary_tokens.update(tokens)
            results.append(" ".join(tokens))

        if not is_query and vocabulary_tokens:
            self._add_to_vocabulary(vocabulary_tokens)

        return results

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Return the spaCy lemma for each input token, preserving order.

        Exposes the lemmatizer in isolation from the rest of the pipeline
        (no stopword removal, no spell correction, no length filtering).
        Multi-word inputs are tokenised by spaCy and their lemmas are
        concatenated with a space, so callers always get exactly one output
        string per input string.
        """
        if not tokens:
            return []
        lemmas: list[str] = []
        for doc in self._nlp.pipe(tokens, batch_size=self.config.batch_size):
            lemmas.append(" ".join(t.lemma_ for t in doc))
        return lemmas

    def _extract_tokens(self, doc: "spacy.tokens.Doc") -> list[str]:
        """Turn a spaCy ``Doc`` into the final list of indexable tokens.

        Performs stopword removal (matched against the surface form, as the
        original pipeline did), lemmatization, and length filtering in a single
        traversal — replacing the previous tokenize → lemmatize double pass.

        Args:
            doc: A spaCy ``Doc`` produced from already-normalized text.

        Returns:
            Lemmatized tokens that are neither stopwords nor out of the
            configured length bounds.
        """
        min_len = self.config.min_token_length
        max_len = self.config.max_token_length
        stopwords = self._stopwords

        tokens: list[str] = []
        for token in doc:
            if token.text in stopwords:
                continue
            lemma = token.lemma_
            if min_len <= len(lemma) <= max_len:
                tokens.append(lemma)
        return tokens

    def _add_to_vocabulary(self, tokens: set[str]) -> None:
        """Add tokens to the spell checker vocabulary.

        Args:
            tokens: Deduplicated processed tokens to insert. The caller
                (:meth:`process_many`) already accumulates a set, so no further
                deduplication is performed here.
        """
        for token in tokens:
            self.spell_checker._insert(token)

    def _correct_spelling(self, tokens: list[str]) -> list[str]:
        """Apply spell correction to tokens using known vocabulary.

        Args:
            tokens: List of tokens to correct.

        Returns:
            List of tokens with spelling corrections applied.
        """
        self._last_corrections = {}
        corrected = []
        for token in tokens:
            correction = self.spell_checker.correct(token)
            result = correction if correction else token
            if result != token:
                self._last_corrections[token] = result
            corrected.append(result)

        return corrected

    def get_last_corrections(self) -> dict[str, str]:
        """Return corrections made by the most recent _correct_spelling call.

        Returns:
            Mapping of original_token -> corrected_token for changed tokens only.
        """
        return dict(self._last_corrections)

    def normalize(self, text: str) -> str:
        """Normalize text by applying lowercase, unicode normalization, and cleaning.

        Applies:
            - Lowercase conversion (if enabled)
            - Unicode normalization (NFC)
            - Accent removal (if enabled)
            - Non-alphanumeric removal (preserves Spanish characters)
            - Multiple spaces collapse

        Args:
            text: Raw text input.

        Returns:
            Normalized text ready for tokenization.
        """
        # Lowercase
        if self.config.lowercase:
            text = text.lower()

        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Optional accent removal
        if self.config.remove_accents:
            text = self._strip_accents(text)

        # Remove non-alphanumeric except Spanish characters
        # Preserves: letters, numbers, spaces, áéíóúüñ
        text = re.sub(r"[^\w\sáéíóúüñ]", " ", text, flags=re.IGNORECASE)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove accents from text.

        Warning: Use with caution for Spanish as it can change meaning
        (año → ano, papá → papa).

        Args:
            text: Text with accents.

        Returns:
            Text with accents removed.
        """
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    @property
    def stopwords(self) -> set[str]:
        """Return the current stopword set (read-only copy)."""
        return self._stopwords.copy()
