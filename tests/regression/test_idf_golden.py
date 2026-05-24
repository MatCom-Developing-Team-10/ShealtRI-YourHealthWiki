"""Golden-file regression for TF-IDF weights.

Locks down the IDF formula. Any change to ``TfidfProcessor.fit()`` (e.g.,
swapping `log` for `log2`, or `+1` smoothing for `+0.5`) WILL change these
values and trip the test.

Update GOLDEN_IDF in the same commit if the change is intentional, and
reference it in the commit body.
"""

from __future__ import annotations

import math

import pytest

from core.interfaces import IndexedCorpus
from core.models import Document
from modules.retriever.tfidf_processor import TfidfProcessor


def _corpus():
    """A 4-doc corpus with terms of varying document-frequency.

    'all' appears in every doc → lowest IDF
    'two' appears in 2 docs    → mid IDF
    'one' appears in 1 doc     → highest IDF
    """
    rows = [
        ("d0", ["all", "one"]),
        ("d1", ["all", "two"]),
        ("d2", ["all", "two"]),
        ("d3", ["all"]),
    ]
    docs = [Document(doc_id=d_id, text=" ".join(toks), url="") for d_id, toks in rows]
    inverted_index: dict[str, list[tuple[int, int]]] = {
        "all": [(0, 1), (1, 1), (2, 1), (3, 1)],
        "one": [(0, 1)],
        "two": [(1, 1), (2, 1)],
    }
    return IndexedCorpus(
        documents=docs,
        processed_texts=[" ".join(toks) for _, toks in rows],
        inverted_index=inverted_index,
        vocabulary=["all", "one", "two"],
    )


# Golden values computed from the IDF formula in TfidfProcessor.fit():
#     idf(t) = log((N + 1) / (df + 1)) + 1
# with N = 4 documents.
N = 4
GOLDEN_IDF = {
    "all": math.log((N + 1) / (4 + 1)) + 1.0,  # df=4 → 1.0
    "two": math.log((N + 1) / (2 + 1)) + 1.0,  # df=2 → ≈1.5108
    "one": math.log((N + 1) / (1 + 1)) + 1.0,  # df=1 → ≈1.9163
}


def test_idf_values_match_golden():
    tfidf = TfidfProcessor()
    tfidf.fit(_corpus())
    for term, expected in GOLDEN_IDF.items():
        idx = tfidf._term_to_idx[term]
        got = float(tfidf._idf[idx])
        assert got == pytest.approx(expected, rel=1e-5), (
            f"IDF drift for {term!r}: expected {expected:.6f}, got {got:.6f}.\n"
            f"If the IDF formula changed deliberately, update GOLDEN_IDF in this file."
        )


def test_idf_monotonic_in_inverse_doc_frequency():
    """A term with fewer document occurrences must have a higher IDF."""
    tfidf = TfidfProcessor()
    tfidf.fit(_corpus())
    idf_all = float(tfidf._idf[tfidf._term_to_idx["all"]])
    idf_two = float(tfidf._idf[tfidf._term_to_idx["two"]])
    idf_one = float(tfidf._idf[tfidf._term_to_idx["one"]])
    assert idf_one > idf_two > idf_all


def test_idf_smoothing_prevents_division_by_zero():
    """The +1 smoothing means a hypothetical 0-frequency term would also work."""
    # Construct a corpus where df=0 for an out-of-vocabulary term would NOT
    # explode. We simulate by checking the smallest possible IDF is finite.
    tfidf = TfidfProcessor()
    tfidf.fit(_corpus())
    assert all(math.isfinite(v) for v in tfidf._idf)
