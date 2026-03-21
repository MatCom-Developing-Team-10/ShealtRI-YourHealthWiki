"""Unit tests for the LSI retriever module."""

from core.models import Document, Query
from modules.retriever import LSIRetriever


def test_lsi_retriever_returns_relevant_medical_document() -> None:
    docs = [
        Document(doc_id="1", text="hypertension symptoms include headache and dizziness"),
        Document(doc_id="2", text="diabetes management includes insulin and diet"),
        Document(doc_id="3", text="asthma causes breathing difficulty and chest tightness"),
    ]

    retriever = LSIRetriever(n_components=2)
    retriever.fit(docs)

    # Includes a typo that should be corrected by the spell checker.
    results = retriever.retrieve(Query(text="hypertensoin symptoms"), top_k=2)

    assert results
    assert results[0].document.doc_id == "1"
    assert results[0].score >= results[1].score
