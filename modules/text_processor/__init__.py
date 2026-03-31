"""Text processor module for preprocessing Spanish medical text."""

from .service import TextProcessor, TextProcessorConfig
from .stopwords import (
    ADDITIONAL_SPANISH_STOPWORDS,
    MEDICAL_ABBREVIATIONS,
    SPANISH_MEDICAL_STOPWORDS,  # Alias for backward compatibility
)

__all__ = [
    "TextProcessor",
    "TextProcessorConfig",
    "ADDITIONAL_SPANISH_STOPWORDS",
    "SPANISH_MEDICAL_STOPWORDS",
    "MEDICAL_ABBREVIATIONS",
]
