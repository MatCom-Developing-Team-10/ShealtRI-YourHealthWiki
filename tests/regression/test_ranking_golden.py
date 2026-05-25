"""Golden-file regression test for the LSI ranker.

A small, hand-curated corpus is indexed deterministically and a fixed set of
queries is issued. The test records the expected top-1 doc_id per query.

The point of this test is to catch *silent* changes in ranking behaviour:
a refactor that flips two documents in the top-3 still satisfies the
semantic-relevance tests (both docs are on-topic) but breaks this golden
test, forcing the author to acknowledge the change.

Mechanics
---------
* Uses ``InMemoryDocumentStore`` + ``InMemoryRepository`` (no ChromaDB).
* Uses the *fake* deterministic TextProcessor from the unit-test layer to
  remove spaCy/Lemma noise — this is regression of the *retrieval logic*,
  not of the lemmatiser.
* If the LSI numerics change deliberately (e.g. switch to a different SVD
  init or a different IDF formula), update GOLDEN_TOP_1 in the same commit
  and reference the change in the commit message.
"""

from __future__ import annotations

from collections import Counter

import pytest

from core.interfaces import IndexedCorpus
from core.models import Document, Query
from modules.retriever import LSIRetriever


# ---------------------------------------------------------------------------
# Fixed corpus (deliberately small so the SVD is fully deterministic).
# ---------------------------------------------------------------------------


_CORPUS = [
    ("d_hta", "hipertensión arterial presión sangre cardiovascular"),
    ("d_dm", "diabetes glucosa insulina páncreas hiperglucemia"),
    ("d_asma", "asma bronquial respiratorio pulmones disnea"),
    ("d_artritis", "artritis articulaciones inflamación autoinmune"),
    ("d_infarto", "infarto miocardio corazón troponina dolor"),
]


def _build_corpus_from_text(rows: list[tuple[str, str]]) -> IndexedCorpus:
    documents = [Document(doc_id=d_id, text=t, url="") for d_id, t in rows]
    processed_texts = [t for _, t in rows]
    inverted_index: dict[str, list[tuple[int, int]]] = {}
    vocabulary_set: set[str] = set()
    for doc_idx, (_, text) in enumerate(rows):
        for term, tf in Counter(text.split()).items():
            inverted_index.setdefault(term, []).append((doc_idx, tf))
            vocabulary_set.add(term)
    return IndexedCorpus(
        documents=documents,
        processed_texts=processed_texts,
        inverted_index=inverted_index,
        vocabulary=sorted(vocabulary_set),
    )


def _query_corpus(text: str) -> IndexedCorpus:
    inv = {term: [(0, tf)] for term, tf in Counter(text.split()).items()}
    return IndexedCorpus(
        documents=[
            Document(
                doc_id="__query__",
                text=text,
                url="",
                metadata={"is_query": True},
            )
        ],
        processed_texts=[text],
        inverted_index=inv,
        vocabulary=sorted(inv.keys()),
    )


# ---------------------------------------------------------------------------
# Golden table: query → expected top-1 doc_id
# ---------------------------------------------------------------------------


GOLDEN_TOP_1 = {
    "presión arterial sangre": "d_hta",
    "diabetes glucosa": "d_dm",
    "asma respiratorio disnea": "d_asma",
    "articulaciones inflamación autoinmune": "d_artritis",
    "infarto corazón dolor troponina": "d_infarto",
}


@pytest.fixture(scope="module")
def fitted_retriever():
    from tests.conftest import InMemoryDocumentStore, InMemoryRepository

    store = InMemoryDocumentStore()
    repo = InMemoryRepository()
    retriever = LSIRetriever(
        repository=repo,
        document_store=store,
        n_components=3,
        similarity_threshold=0.0,
    )
    corpus = _build_corpus_from_text(_CORPUS)
    retriever.fit(corpus)
    return retriever


@pytest.mark.parametrize("query,expected_id", list(GOLDEN_TOP_1.items()))
def test_golden_top_1_per_query(fitted_retriever, query, expected_id):
    """Ranking output for the golden corpus must not drift silently."""
    results = fitted_retriever.retrieve(
        Query(text=query, indexed_corpus=_query_corpus(query)),
        top_k=1,
    )
    assert results, f"no results for golden query {query!r}"
    got = results[0].document.doc_id
    assert got == expected_id, (
        f"Golden mismatch for {query!r}\n"
        f"  expected: {expected_id}\n"
        f"  got:      {got}\n"
        f"If this change is intentional, update GOLDEN_TOP_1 in this file."
    )


def test_score_monotonicity_for_repeated_terms(fitted_retriever):
    """Adding more on-topic terms to a query must not decrease the top score.

    This protects against bugs where IDF or normalisation accidentally
    penalise longer queries.
    """
    short_results = fitted_retriever.retrieve(
        Query(text="diabetes", indexed_corpus=_query_corpus("diabetes")),
        top_k=1,
    )
    long_results = fitted_retriever.retrieve(
        Query(
            text="diabetes glucosa insulina",
            indexed_corpus=_query_corpus("diabetes glucosa insulina"),
        ),
        top_k=1,
    )
    short_score = short_results[0].score if short_results else 0.0
    long_score = long_results[0].score if long_results else 0.0
    assert long_score >= short_score, (
        f"longer on-topic query lost score: short={short_score:.4f} long={long_score:.4f}"
    )
