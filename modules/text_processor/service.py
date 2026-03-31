"""Text preprocessing pipeline for Spanish medical documents using NLTK.

This module provides text preprocessing for the LSI indexing pipeline:
    Raw text → normalize → tokenize → remove stopwords → stem → filtered tokens

Usage:
    processor = TextProcessor()
    processed = processor.process("La hipertensión arterial causa cefalea")
    # Returns: "hipertens arter caus cefale"
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from .stopwords import SPANISH_MEDICAL_STOPWORDS


logger = logging.getLogger(__name__)


@dataclass
class TextProcessorConfig:
    """Configuration for text preprocessing.

    Attributes:
        language: Language for stemming and stopwords (default: "spanish").
        min_token_length: Minimum token length to keep (default: 2).
        max_token_length: Maximum token length to keep (default: 50).
        remove_accents: Whether to normalize accented characters (default: False).
            Keep False for Spanish to preserve meaning (año vs ano).
        lowercase: Whether to convert to lowercase (default: True).
        custom_stopwords: Additional stopwords to remove beyond defaults.
    """

    language: str = "spanish"
    min_token_length: int = 2
    max_token_length: int = 20
    remove_accents: bool = False
    lowercase: bool = True
    custom_stopwords: set[str] = field(default_factory=set)


class TextProcessor:
    """Preprocesses Spanish medical text for LSI indexing.

    Pipeline stages:
        1. Normalization (lowercase, unicode normalization)
        2. Tokenization (NLTK word_tokenize for Spanish)
        3. Stopword removal (NLTK Spanish + medical domain)
        4. Stemming (SnowballStemmer for Spanish)
        5. Token filtering (length constraints)

    Example:
        processor = TextProcessor()
        result = processor.process("La hipertensión arterial causa dolores de cabeza")
        # Result: "hipertens arter caus dolor cabez"

        # With custom config
        config = TextProcessorConfig(min_token_length=3)
        processor = TextProcessor(config)
    """

    def __init__(self, config: TextProcessorConfig | None = None) -> None:
        """Initialize the text processor.

        Args:
            config: Processing configuration. Uses defaults if None.

        Note:
            Downloads required NLTK data on first use if not present.
        """
        self.config = config or TextProcessorConfig()
        self._ensure_nltk_data()

        self._stemmer = SnowballStemmer(self.config.language)
        self._stopwords = self._load_stopwords()

        logger.debug(
            f"TextProcessor initialized: language={self.config.language}, "
            f"stopwords={len(self._stopwords)}"
        )

    def _ensure_nltk_data(self) -> None:
        """Download NLTK data packages if not present."""
        required_packages = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
        ]

        for path, package in required_packages:
            try:
                nltk.data.find(path)
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)

    def _load_stopwords(self) -> set[str]:
        """Load Spanish stopwords plus custom domain stopwords."""
        from nltk.corpus import stopwords as nltk_stopwords

        base_stopwords = set(nltk_stopwords.words(self.config.language))
        combined = base_stopwords | SPANISH_MEDICAL_STOPWORDS | self.config.custom_stopwords

        logger.debug(
            f"Loaded stopwords: {len(base_stopwords)} NLTK + "
            f"{len(SPANISH_MEDICAL_STOPWORDS)} medical + "
            f"{len(self.config.custom_stopwords)} custom"
        )

        return combined

    def process(self, text: str) -> str:
        """Apply the full preprocessing pipeline to a text.

        Args:
            text: Raw input text.

        Returns:
            Space-joined preprocessed tokens ready for TF-IDF.
        """
        if not text or not text.strip():
            return ""

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        tokens = self.filter_tokens(tokens)

        return " ".join(tokens)

    def tokenize(self, text: str) -> list[str]:
        """Normalize and tokenize text.

        Applies:
            - Lowercase conversion (if enabled)
            - Unicode normalization (NFC)
            - Accent removal (if enabled)
            - Non-alphanumeric removal (preserves Spanish characters)
            - NLTK word tokenization

        Args:
            text: Raw text input.

        Returns:
            List of normalized tokens.
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

        # NLTK tokenization for Spanish
        try:
            tokens = word_tokenize(text, language=self.config.language)
        except LookupError:
            # Fallback to simple split if NLTK data not available
            logger.warning("NLTK tokenizer not available, using simple split")
            tokens = text.split()

        return tokens

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords from token list.

        Args:
            tokens: List of tokens.

        Returns:
            Filtered token list without stopwords.
        """
        return [t for t in tokens if t not in self._stopwords]

    def stem(self, tokens: list[str]) -> list[str]:
        """Apply stemming to tokens.

        Uses NLTK SnowballStemmer for Spanish which handles:
            - medicamentos → medic
            - hipertensión → hipertens
            - arterial → arter

        Args:
            tokens: List of tokens.

        Returns:
            List of stemmed tokens.
        """
        return [self._stemmer.stem(t) for t in tokens]

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
