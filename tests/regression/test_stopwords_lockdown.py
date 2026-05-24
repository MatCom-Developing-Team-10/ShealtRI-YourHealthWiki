"""Regression test for stopword vocabularies.

Stopword lists are *configuration* — a silent change to them changes the
retrieval results of every query. This test snapshots the current size of
each list and the membership of a hand-curated representative subset.

When a stopword is added or removed intentionally, update SNAPSHOT_SIZES
and the membership assertions in the same commit, and reference the
change in the commit message.
"""

from __future__ import annotations

from modules.text_processor.stopwords import (
    ADDITIONAL_SPANISH_STOPWORDS,
    MEDICAL_ABBREVIATIONS,
    SPANISH_MEDICAL_STOPWORDS,
)


# Recorded the day the regression test was authored. Use a tolerance instead
# of an exact match so trivial typo fixes don't trip the test, but anything
# beyond +-3 entries means somebody changed the set deliberately.
SNAPSHOT_SIZES = {
    "ADDITIONAL_SPANISH_STOPWORDS": 83,
    "MEDICAL_ABBREVIATIONS": 46,
}


def test_alias_identity():
    """SPANISH_MEDICAL_STOPWORDS must remain an alias for ADDITIONAL_SPANISH_STOPWORDS."""
    assert SPANISH_MEDICAL_STOPWORDS is ADDITIONAL_SPANISH_STOPWORDS


def test_stopwords_size_within_tolerance():
    actual = len(ADDITIONAL_SPANISH_STOPWORDS)
    expected = SNAPSHOT_SIZES["ADDITIONAL_SPANISH_STOPWORDS"]
    assert abs(actual - expected) <= 3, (
        f"Stopword set size changed unexpectedly: {expected} → {actual}.\n"
        f"Update SNAPSHOT_SIZES if the change was intentional."
    )


def test_medical_abbreviations_size_within_tolerance():
    actual = len(MEDICAL_ABBREVIATIONS)
    expected = SNAPSHOT_SIZES["MEDICAL_ABBREVIATIONS"]
    assert abs(actual - expected) <= 3, (
        f"Medical abbreviation set size changed unexpectedly: {expected} → {actual}.\n"
        f"Update SNAPSHOT_SIZES if the change was intentional."
    )


def test_no_overlap_between_stopwords_and_medical_abbreviations():
    """Critical invariant. If an abbreviation is also a stopword it gets dropped."""
    overlap = ADDITIONAL_SPANISH_STOPWORDS & MEDICAL_ABBREVIATIONS
    assert overlap == set(), (
        f"Overlap between stopwords and medical abbreviations: {overlap}. "
        f"This would silently drop these terms from indexed documents."
    )


def test_medical_terms_are_NOT_stopwords():
    """Hand-curated list of terms that MUST survive preprocessing."""
    must_survive = {
        "paciente",
        "tratamiento",
        "síntoma",
        "diagnóstico",
        "enfermedad",
        "medicamento",
        "alto",
        "bajo",
        "mayor",
        "menor",
        "agudo",
        "crónico",
    }
    leaked = must_survive & ADDITIONAL_SPANISH_STOPWORDS
    assert leaked == set(), (
        f"Medical-relevant terms accidentally added to stopwords: {leaked}"
    )


def test_known_abbreviations_present():
    """Hand-curated list of abbreviations that MUST stay preserved."""
    required = {"hta", "dm", "dm2", "epoc", "iam", "acv", "ecg"}
    missing = required - MEDICAL_ABBREVIATIONS
    assert missing == set(), (
        f"Critical medical abbreviations removed: {missing}"
    )
