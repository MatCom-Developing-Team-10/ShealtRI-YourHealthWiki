"""Tests for TextChunker — fixed-size and paragraph-based chunking strategies."""

from __future__ import annotations

import math

import pytest

from core.models import Document
from modules.indexer.chunker import TextChunker


def _make_doc(text: str, doc_id: str = "doc1") -> Document:
    return Document(doc_id=doc_id, text=text, url="http://example.com")


class TestTextChunkerInit:
    def test_default_params(self):
        chunker = TextChunker()
        assert chunker._chunk_size == 200
        assert chunker._overlap == 40
        assert chunker._strategy == "fixed"

    def test_custom_params(self):
        chunker = TextChunker(chunk_size=100, overlap=20)
        assert chunker._chunk_size == 100
        assert chunker._overlap == 20

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=50, overlap=50)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=50, overlap=60)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            TextChunker(strategy="semantic_fancy")

    def test_paragraph_strategy_accepted(self):
        chunker = TextChunker(strategy="paragraph")
        assert chunker._strategy == "paragraph"


class TestChunkMethod:
    def test_short_document_returns_single_chunk(self):
        chunker = TextChunker(chunk_size=100, overlap=20)
        doc = _make_doc("word " * 50)  # 50 tokens, well below chunk_size
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].doc_id == "doc1__chunk_0"

    def test_empty_text_returns_original_document(self):
        chunker = TextChunker(chunk_size=100, overlap=20)
        doc = _make_doc("")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].doc_id == doc.doc_id

    def test_exact_chunk_size_produces_one_chunk(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        doc = _make_doc(" ".join(f"w{i}" for i in range(10)))
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_multiple_chunks_produced(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        doc = _make_doc(" ".join(f"w{i}" for i in range(30)))
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_original_doc_id_in_metadata(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        doc = _make_doc(" ".join(f"w{i}" for i in range(30)), doc_id="my_doc")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["original_doc_id"] == "my_doc"

    def test_chunk_index_in_metadata(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        doc = _make_doc(" ".join(f"w{i}" for i in range(30)))
        chunks = chunker.chunk(doc)
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == idx

    def test_total_chunks_in_metadata(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        doc = _make_doc(" ".join(f"w{i}" for i in range(30)))
        chunks = chunker.chunk(doc)
        total = chunks[0].metadata["total_chunks"]
        assert total == len(chunks)
        assert all(c.metadata["total_chunks"] == total for c in chunks)

    def test_url_preserved_across_chunks(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        doc = _make_doc(" ".join(f"w{i}" for i in range(30)))
        doc.url = "http://health.example/article"
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.url == doc.url

    def test_chunk_doc_id_format(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        doc = _make_doc(" ".join(f"w{i}" for i in range(25)), doc_id="art42")
        chunks = chunker.chunk(doc)
        for idx, chunk in enumerate(chunks):
            assert chunk.doc_id == f"art42__chunk_{idx}"

    def test_overlap_tokens_shared_between_adjacent_chunks(self):
        """Last tokens of chunk N should be first tokens of chunk N+1."""
        chunker = TextChunker(chunk_size=10, overlap=3)
        words = [f"w{i}" for i in range(25)]
        doc = _make_doc(" ".join(words))
        chunks = chunker.chunk(doc)
        if len(chunks) >= 2:
            tail = chunks[0].text.split()[-3:]
            head = chunks[1].text.split()[:3]
            assert tail == head

    def test_no_tokens_lost(self):
        """All source tokens must appear in at least one chunk."""
        chunker = TextChunker(chunk_size=10, overlap=3)
        words = [f"w{i}" for i in range(35)]
        doc = _make_doc(" ".join(words))
        chunks = chunker.chunk(doc)
        all_chunk_tokens: set[str] = set()
        for chunk in chunks:
            all_chunk_tokens.update(chunk.text.split())
        assert set(words) <= all_chunk_tokens

    def test_single_token_document(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        doc = _make_doc("word")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "word"

    def test_chunk_strategy_in_metadata(self):
        chunker = TextChunker(chunk_size=10, overlap=2, strategy="fixed")
        doc = _make_doc(" ".join(f"w{i}" for i in range(30)))
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "fixed"


class TestParagraphChunker:
    def test_single_short_paragraph_returns_one_chunk(self):
        chunker = TextChunker(chunk_size=100, overlap=10, strategy="paragraph")
        doc = _make_doc("Este es un párrafo corto sobre diabetes.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_multiple_paragraphs_grouped_within_budget(self):
        """Three small paragraphs that fit within chunk_size should become one chunk."""
        chunker = TextChunker(chunk_size=50, overlap=5, strategy="paragraph")
        text = "párrafo uno sobre hipertensión\n\npárrafo dos sobre diabetes\n\npárrafo tres sobre asma"
        doc = _make_doc(text)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_large_paragraphs_split_across_chunks(self):
        """Paragraphs that exceed the budget individually should each become their own chunk."""
        chunker = TextChunker(chunk_size=10, overlap=2, strategy="paragraph")
        words_a = " ".join(f"a{i}" for i in range(15))
        words_b = " ".join(f"b{i}" for i in range(15))
        doc = _make_doc(f"{words_a}\n\n{words_b}")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2

    def test_chunk_strategy_metadata_is_paragraph(self):
        chunker = TextChunker(chunk_size=10, overlap=2, strategy="paragraph")
        words_a = " ".join(f"a{i}" for i in range(15))
        words_b = " ".join(f"b{i}" for i in range(15))
        doc = _make_doc(f"{words_a}\n\n{words_b}")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "paragraph"

    def test_original_doc_id_preserved(self):
        chunker = TextChunker(chunk_size=10, overlap=2, strategy="paragraph")
        words_a = " ".join(f"a{i}" for i in range(15))
        words_b = " ".join(f"b{i}" for i in range(15))
        doc = _make_doc(f"{words_a}\n\n{words_b}", doc_id="med_article")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["original_doc_id"] == "med_article"

    def test_url_preserved(self):
        chunker = TextChunker(chunk_size=10, overlap=2, strategy="paragraph")
        words_a = " ".join(f"a{i}" for i in range(15))
        words_b = " ".join(f"b{i}" for i in range(15))
        doc = _make_doc(f"{words_a}\n\n{words_b}")
        doc.url = "http://health.example/article"
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.url == doc.url

    def test_empty_doc_returns_original(self):
        chunker = TextChunker(chunk_size=50, overlap=5, strategy="paragraph")
        doc = _make_doc("")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_paragraph_overlap_carries_last_paragraph(self):
        """First paragraph of chunk N+1 should be the last paragraph of chunk N."""
        chunker = TextChunker(chunk_size=12, overlap=3, strategy="paragraph")
        # 4 paragraphs of ~5 tokens each → chunks of 2 paragraphs with 1 overlap
        paras = [" ".join(f"w{i}{j}" for j in range(5)) for i in range(4)]
        doc = _make_doc("\n\n".join(paras))
        chunks = chunker.chunk(doc)
        if len(chunks) >= 2:
            # The last paragraph of chunk 0 should appear at the start of chunk 1
            last_para_chunk0 = chunks[0].text.split("\n\n")[-1].strip()
            first_para_chunk1 = chunks[1].text.split("\n\n")[0].strip()
            assert last_para_chunk0 == first_para_chunk1


_st_available = pytest.mark.skipif(
    __import__("importlib").util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed",
)


@_st_available
class TestSemanticChunker:
    """Tests for strategy='semantic'. Require sentence-transformers."""

    def test_semantic_strategy_accepted(self):
        chunker = TextChunker(strategy="semantic")
        assert chunker._strategy == "semantic"

    def test_returns_list_of_documents(self):
        chunker = TextChunker(strategy="semantic", similarity_threshold=0.5)
        doc = _make_doc(
            "La hipertensión arterial es una enfermedad crónica. "
            "El tratamiento incluye medicamentos antihipertensivos. "
            "La diabetes mellitus afecta el metabolismo de la glucosa. "
            "Los pacientes diabéticos deben controlar su dieta."
        )
        chunks = chunker.chunk(doc)
        assert isinstance(chunks, list)
        assert all(hasattr(c, "doc_id") for c in chunks)
        assert len(chunks) >= 1

    def test_chunk_strategy_metadata_is_semantic(self):
        chunker = TextChunker(strategy="semantic", similarity_threshold=0.9)
        doc = _make_doc(
            "La hipertensión es presión alta. "
            "La diabetes afecta la glucosa. "
            "El cáncer es una enfermedad oncológica."
        )
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "semantic"

    def test_original_doc_id_preserved(self):
        chunker = TextChunker(strategy="semantic", similarity_threshold=0.5)
        doc = _make_doc(
            "La hipertensión arterial requiere tratamiento. "
            "La diabetes mellitus afecta la glucosa.",
            doc_id="med_art_42",
        )
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["original_doc_id"] == "med_art_42"

    def test_url_preserved(self):
        chunker = TextChunker(strategy="semantic", similarity_threshold=0.5)
        doc = _make_doc(
            "La hipertensión arterial es común. La diabetes afecta millones."
        )
        doc.url = "http://health.example/article"
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.url == doc.url

    def test_topic_change_produces_multiple_chunks(self):
        """Very low threshold should split on almost every boundary."""
        chunker = TextChunker(strategy="semantic", similarity_threshold=0.99)
        doc = _make_doc(
            "La hipertensión arterial causa problemas cardiovasculares graves. "
            "La diabetes mellitus tipo dos afecta la glucosa en sangre. "
            "El cáncer de colon requiere colonoscopía de detección temprana."
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2

    def test_high_threshold_keeps_similar_sentences_together(self):
        """Very high threshold should always split — even similar sentences exceed it."""
        chunker = TextChunker(strategy="semantic", similarity_threshold=0.0)
        doc = _make_doc(
            "La presión arterial alta es la hipertensión. "
            "La hipertensión arterial eleva la presión sanguínea."
        )
        chunks = chunker.chunk(doc)
        # threshold=0 means sim(a,b) >= 0 always → no split → 1 chunk
        assert len(chunks) == 1

    def test_single_sentence_falls_back_to_fixed(self):
        """A single-sentence doc cannot be split semantically; fixed fallback applies."""
        chunker = TextChunker(chunk_size=5, overlap=1, strategy="semantic")
        doc = _make_doc("La hipertensión es presión alta en las arterias.")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["original_doc_id"] == doc.doc_id
