"""Tests for the stopword vocabularies.

These are configuration constants, but contract tests guard against silent
regressions (e.g., a typo merging the two sets, an alias breaking, or a
medical abbreviation accidentally being added as a stopword).
"""

from __future__ import annotations

from modules.text_processor.stopwords import (
    ADDITIONAL_SPANISH_STOPWORDS,
    MEDICAL_ABBREVIATIONS,
    SPANISH_MEDICAL_STOPWORDS,
)


class TestAlias:
    def test_legacy_alias_is_same_set(self):
        # Backward-compatibility alias must point to the same object.
        assert SPANISH_MEDICAL_STOPWORDS is ADDITIONAL_SPANISH_STOPWORDS


class TestStopwordsContent:
    def test_contains_filler_word(self):
        assert "etc" in ADDITIONAL_SPANISH_STOPWORDS
        assert "etcétera" in ADDITIONAL_SPANISH_STOPWORDS

    def test_contains_generic_qualifier(self):
        # These should be removed since they don't add medical value
        assert "mismo" in ADDITIONAL_SPANISH_STOPWORDS
        assert "varios" in ADDITIONAL_SPANISH_STOPWORDS

    def test_does_not_contain_medical_term(self):
        # Medical terms must NOT be filtered as stopwords
        for term in ("paciente", "tratamiento", "síntoma", "alto", "bajo"):
            assert term not in ADDITIONAL_SPANISH_STOPWORDS, term

    def test_no_overlap_with_medical_abbreviations(self):
        # If a stopword and a medical abbreviation overlap, retrieval breaks.
        overlap = ADDITIONAL_SPANISH_STOPWORDS & MEDICAL_ABBREVIATIONS
        assert overlap == set(), f"overlap: {overlap}"


class TestMedicalAbbreviations:
    def test_cardiovascular_abbreviations(self):
        for abbr in ("hta", "iam", "icc", "ecg", "ekg", "acv", "fa"):
            assert abbr in MEDICAL_ABBREVIATIONS, abbr

    def test_metabolic_abbreviations(self):
        for abbr in ("dm", "dm1", "dm2", "hba1c", "imc"):
            assert abbr in MEDICAL_ABBREVIATIONS, abbr

    def test_respiratory_abbreviations(self):
        for abbr in ("epoc", "covid", "asma"):
            assert abbr in MEDICAL_ABBREVIATIONS, abbr

    def test_no_empty_strings(self):
        assert "" not in ADDITIONAL_SPANISH_STOPWORDS
        assert "" not in MEDICAL_ABBREVIATIONS
