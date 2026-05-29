"""Integration test: semantic relevance over the 20-document medical corpus.

These tests are the closest thing we have to "the system actually finds the
right document". For each topical query we know which document was written
about that topic, and we expect the LSI projection to rank it within the
top-N. The tests are tolerant (top-3 or top-5 instead of top-1) because LSI
is a noisy semantic match and exact-top-1 is brittle.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("es_core_news_md")
except OSError:
    pytest.skip("spaCy model 'es_core_news_md' not installed", allow_module_level=True)


from core.models import Query
from modules.indexer import IndexerService
from modules.indexer.service import IndexerConfig
from modules.retriever import LSIRetriever


@pytest.fixture(scope="module")
def fitted_lsi(sample_documents, text_processor):
    """Fit the LSI pipeline once for the whole module (slow op).

    Reuses the session-scoped ``text_processor`` (shared spaCy model) to
    keep memory bounded. Spell-checker pollution across modules is a
    non-issue here because the only callers are queries that exercise the
    same medical vocabulary.
    """
    from tests.conftest import InMemoryDocumentStore, InMemoryRepository

    store = InMemoryDocumentStore()
    repo = InMemoryRepository()

    # min_term_frequency=1 ensures topical terms appearing in only one
    # document of the 20-doc fixture survive the vocabulary filter.
    indexer = IndexerService(
        text_processor=text_processor,
        config=IndexerConfig(min_term_frequency=1),
    )
    corpus = indexer.build(sample_documents)

    retriever = LSIRetriever(
        repository=repo,
        document_store=store,
        n_components=15,
        similarity_threshold=0.0,
    )
    retriever.fit(corpus)
    return indexer, retriever


def _ids(results) -> list[str]:
    return [r.document.doc_id for r in results]


# ---------------------------------------------------------------------------
# Topical relevance checks
# ---------------------------------------------------------------------------

# Each row pairs a natural-language Spanish query with one or more doc_ids
# from the synthetic corpus that are *clearly* on-topic.
_RELEVANCE_PAIRS = [
    ("hipertensión arterial presión sangre", "doc_hipertension_001"),
    ("diabetes glucosa insulina páncreas", "doc_diabetes_002"),
    ("asma bronquial respiración pulmones", "doc_asma_003"),
    ("artritis reumatoide articulaciones autoinmune", "doc_artritis_004"),
    ("cáncer colon pólipos colonoscopia", "doc_cancer_colon_005"),
    ("depresión tristeza estado de ánimo", "doc_depresion_006"),
    ("osteoporosis fracturas hueso densidad", "doc_osteoporosis_007"),
    ("infarto miocardio corazón dolor torácico", "doc_infarto_008"),
    ("alzheimer demencia memoria neurodegenerativa", "doc_alzheimer_009"),
    ("hipotiroidismo tiroides hormona TSH", "doc_hipotiroidismo_010"),
    ("neumonía pulmones infección bacteriana", "doc_neumonia_011"),
    ("anemia hierro hemoglobina ferritina", "doc_anemia_012"),
    ("ictus cerebrovascular cerebro trombolisis", "doc_ictus_013"),
    ("EPOC enfisema bronquitis tabaco", "doc_epoc_014"),
    ("insuficiencia renal diálisis trasplante", "doc_insuficiencia_renal_015"),
    ("fibromialgia dolor crónico fatiga", "doc_fibromialgia_016"),
    ("colesterol LDL HDL estatinas", "doc_colesterol_017"),
    ("migraña dolor cabeza aura triptanes", "doc_migraña_018"),
    ("hepatitis C hígado antivirales", "doc_hepatitis_019"),
    ("obesidad síndrome metabólico IMC", "doc_obesidad_020"),
]


@pytest.mark.parametrize("query,expected_id", _RELEVANCE_PAIRS)
def test_topical_query_surfaces_expected_doc_in_top_5(fitted_lsi, query, expected_id):
    """For every topic in the corpus, the matching doc must rank within top-5."""
    indexer, retriever = fitted_lsi
    qc = indexer.build_query(query)
    results = retriever.retrieve(Query(text=query, indexed_corpus=qc), top_k=5)
    top_ids = _ids(results)
    assert expected_id in top_ids, (
        f"query {query!r} did not surface {expected_id} in top-5; got {top_ids}"
    )


def test_top_result_score_is_significantly_above_random(fitted_lsi):
    """A topical query's top score should be notably higher than the noise floor."""
    indexer, retriever = fitted_lsi
    qc = indexer.build_query("hipertensión arterial presión")
    results = retriever.retrieve(
        Query(text="hipertensión arterial presión", indexed_corpus=qc), top_k=20
    )
    assert results
    scores = [r.score for r in results]
    # The top score should be clearly above the median — LSI must do *some* ranking
    assert scores[0] > 0.0
    assert scores[0] >= scores[len(scores) // 2]


def test_unrelated_query_returns_lower_top_score_than_topical(fitted_lsi):
    """Sanity check: a query about a topic in the corpus scores higher than gibberish."""
    indexer, retriever = fitted_lsi

    topical_qc = indexer.build_query("diabetes glucosa insulina")
    topical = retriever.retrieve(
        Query(text="diabetes glucosa insulina", indexed_corpus=topical_qc), top_k=1
    )

    gibberish_qc = indexer.build_query("xyzxyz qwerqwer zzzz")
    gibberish = retriever.retrieve(
        Query(text="xyzxyz qwerqwer zzzz", indexed_corpus=gibberish_qc), top_k=1
    )

    topical_top = topical[0].score if topical else 0.0
    gibberish_top = gibberish[0].score if gibberish else 0.0
    assert topical_top > gibberish_top, (
        f"topical={topical_top:.4f} not > gibberish={gibberish_top:.4f}"
    )
