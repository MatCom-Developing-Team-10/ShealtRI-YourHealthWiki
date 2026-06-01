"""Regression tests for the medical thesaurus.

Locks down:
    - Size of the thesaurus map (±5 tolerance).
    - A small set of MUST-have edges that real queries depend on.
    - Bidirectionality invariant — every (a -> b) must imply (b -> a).
    - No accidental collisions with the project's stopword set, which would
      cause expansion to inject terms that get dropped immediately.
"""

from __future__ import annotations

from core.stopwords import ADDITIONAL_SPANISH_STOPWORDS
from plugins.expansion.thesaurus import MEDICAL_THESAURUS, expand_term


# Snapshot the day this regression test was authored. Tolerance allows
# typo fixes and minor curation; anything outside the band means a
# deliberate change that should be acknowledged.
SNAPSHOT_SIZE = 109
TOLERANCE = 5


# Edges every real lay/technical query depends on. Removing any of these
# silently breaks the integration test in tests/integration/test_query_expansion.py.
GOLDEN_EDGES = [
    ("hipertensión", "hta"),
    ("hta", "hipertensión"),
    ("ataque", "infarto"),
    ("infarto", "ataque"),
    ("azúcar", "glucosa"),
    ("glucosa", "azúcar"),
    ("diabetes", "dm"),
    ("dm", "diabetes"),
    ("ictus", "acv"),
    ("acv", "ictus"),
    ("alzheimer", "demencia"),
    ("migraña", "cefalea"),
]


class TestSizeSnapshot:
    def test_size_within_tolerance(self):
        actual = len(MEDICAL_THESAURUS)
        delta = abs(actual - SNAPSHOT_SIZE)
        assert delta <= TOLERANCE, (
            f"Thesaurus size changed by more than ±{TOLERANCE}: "
            f"snapshot={SNAPSHOT_SIZE}, actual={actual}.\n"
            f"If this change is intentional, update SNAPSHOT_SIZE in this file."
        )


class TestGoldenEdges:
    def test_every_golden_edge_present(self):
        missing = [
            (src, dst)
            for src, dst in GOLDEN_EDGES
            if dst not in expand_term(src)
        ]
        assert missing == [], (
            f"Golden thesaurus edges missing: {missing}.\n"
            f"These edges are required by the integration test suite. "
            f"Restore them in plugins/expansion/thesaurus.py."
        )


class TestBidirectionalityInvariant:
    def test_every_edge_is_bidirectional(self):
        """Every (a, b) edge must be matched by a (b, a) edge."""
        broken: list[tuple[str, str]] = []
        for term, synonyms in MEDICAL_THESAURUS.items():
            for syn in synonyms:
                if term not in MEDICAL_THESAURUS.get(syn, frozenset()):
                    broken.append((term, syn))
        assert broken == [], (
            f"Bidirectionality broken for: {broken}.\n"
            f"_build_bidirectional() in plugins/expansion/thesaurus.py "
            f"must symmetrise every edge."
        )


class TestNoStopwordCollisions:
    def test_no_thesaurus_term_is_a_stopword(self):
        """An expansion candidate must not be in the project's stopword set;
        otherwise the indexer would drop it immediately, defeating expansion.
        """
        collisions = (
            set(MEDICAL_THESAURUS.keys()) & ADDITIONAL_SPANISH_STOPWORDS
        )
        # Also check synonym values
        for synonyms in MEDICAL_THESAURUS.values():
            collisions |= set(synonyms) & ADDITIONAL_SPANISH_STOPWORDS

        assert collisions == set(), (
            f"Thesaurus terms collide with stopwords: {collisions}. "
            f"These terms would be silently dropped by TextProcessor; "
            f"either remove them from the thesaurus or remove them from "
            f"ADDITIONAL_SPANISH_STOPWORDS (whichever is correct for the domain)."
        )
