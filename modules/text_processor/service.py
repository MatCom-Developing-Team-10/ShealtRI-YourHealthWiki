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

from .spell_checker import TrieSpellChecker
from .stopwords import SPANISH_MEDICAL_STOPWORDS


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
    """

    language: str = "spanish"
    spacy_model: str = "es_core_news_md"
    min_token_length: int = 2
    max_token_length: int = 20
    remove_accents: bool = False
    lowercase: bool = True
    custom_stopwords: set[str] = field(default_factory=set)


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

    def process(self, text: str, is_query: bool = False) -> str:
        """Apply the full preprocessing pipeline to a text.

        Pipeline flow:
            1. Normalize (lowercase, unicode, cleaning)
            2. Tokenize (split into tokens)
            3. Remove stopwords
            4. Lemmatize (reduce to base form)
            5. Filter tokens (length constraints)
            6. If is_query=False: add tokens to spell checker vocabulary
               If is_query=True: correct tokens using known vocabulary

        Args:
            text: Raw input text.
            is_query: If False, tokens are added to spell checker vocabulary.
                      If True, tokens are corrected using the vocabulary.

        Returns:
            Space-joined preprocessed tokens ready for indexer.
        """
        if not text or not text.strip():
            return ""

        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        tokens = self.filter_tokens(tokens)

        if is_query:
            # Correct tokens using known vocabulary
            tokens = self._correct_spelling(tokens)
        else:
            # Add tokens to vocabulary for future corrections
            self._add_to_vocabulary(tokens)

        return " ".join(tokens)

    def _add_to_vocabulary(self, tokens: list[str]) -> None:
        """Add tokens to the spell checker vocabulary.

        Args:
            tokens: List of processed tokens to add.
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
        corrected = []
        for token in tokens:
            correction = self.spell_checker.correct(token)
            # Use correction if found, otherwise keep the original token
            corrected.append(correction if correction else token)

        return corrected

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

    def tokenize(self, text: str) -> list[str]:
        """Tokenize normalized text using spaCy.

        Args:
            text: Normalized text input.

        Returns:
            List of tokens.
        """
        doc = self._nlp(text)
        return [token.text for token in doc]

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords from token list.

        Args:
            tokens: List of tokens.

        Returns:
            Filtered token list without stopwords.
        """
        return [t for t in tokens if t not in self._stopwords]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Apply lemmatization to tokens using spaCy.

        Uses spaCy's lemmatizer with POS tagging for accurate lemmatization:
            - medicamentos → medicamento
            - hipertensión → hipertensión (preserved)
            - arterial → arterial (preserved as adjective)
            - causa → causar (verb infinitive)

        Args:
            tokens: List of tokens.

        Returns:
            List of lemmatized tokens.

        Note:
            Unlike stemming, lemmatization preserves valid word forms and is
            more suitable for medical terminology where precision matters.
        """
        # Process tokens as a single document for POS tagging context
        text = " ".join(tokens)
        doc = self._nlp(text)
        return [token.lemma_ for token in doc]

    def filter_tokens(self, tokens: list[str]) -> list[str]:
        """Filter tokens by length constraints.

        Removes tokens that are:
            - Too short (< min_token_length)
            - Too long (> max_token_length)

        Args:
            tokens: List of tokens.

        Returns:
            Filtered token list.
        """
        return [
            t for t in tokens
            if self.config.min_token_length <= len(t) <= self.config.max_token_length
        ]

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
