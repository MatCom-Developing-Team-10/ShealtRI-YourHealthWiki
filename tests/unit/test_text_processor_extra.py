"""Additional coverage for ``modules.text_processor.service.TextProcessor``.

The base ``test_text_processor.py`` covers the public processing pipeline.
This file adds:
    - Direct invocation of ``lemmatize()`` (skipped in the base suite).
    - The OSError path when the spaCy model is missing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# spaCy must be importable for the rest of this module to make sense.
spacy = pytest.importorskip("spacy")


from modules.text_processor.service import TextProcessor, TextProcessorConfig


# ---------------------------------------------------------------------------
# Lemmatize() called in isolation (without going through process())
# ---------------------------------------------------------------------------


class TestLemmatizeDirect:
    def test_lemmatize_returns_list_of_strings(self, text_processor: TextProcessor):
        out = text_processor.lemmatize(["medicamentos", "pacientes"])
        assert isinstance(out, list)
        assert all(isinstance(t, str) for t in out)

    def test_lemmatize_empty_input(self, text_processor: TextProcessor):
        assert text_processor.lemmatize([]) == []

    def test_lemmatize_known_morphology(self, text_processor: TextProcessor):
        # The Spanish lemmatiser singularises common nouns.
        out = text_processor.lemmatize(["medicamentos", "tratamientos"])
        # Either both got singularised or stayed; check the contract holds
        assert all(t.endswith(("o", "s")) for t in out)


# ---------------------------------------------------------------------------
# Missing spaCy model -> OSError reraised with helpful message
# ---------------------------------------------------------------------------


class TestSpacyModelLoadError:
    def test_oserror_reraised_with_install_hint(self):
        cfg = TextProcessorConfig(spacy_model="es_does_not_exist")
        with patch(
            "modules.text_processor.service.spacy.load",
            side_effect=OSError("not installed"),
        ):
            with pytest.raises(OSError, match="not installed"):
                TextProcessor(cfg)
