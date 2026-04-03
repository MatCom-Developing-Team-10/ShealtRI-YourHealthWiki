"""Text processor module for preprocessing Spanish medical text."""

from .service import TextProcessor, TextProcessorConfig
from .spell_checker import TrieSpellChecker
from .stopwords import (
    ADDITIONAL_SPANISH_STOPWORDS,
    MEDICAL_ABBREVIATIONS,
    SPANISH_MEDICAL_STOPWORDS,  # Alias for backward compatibility
)

__all__ = [
    "TextProcessor",
    "TextProcessorConfig",
    "TrieSpellChecker",
    "ADDITIONAL_SPANISH_STOPWORDS",
    "SPANISH_MEDICAL_STOPWORDS",
    "MEDICAL_ABBREVIATIONS",
]
