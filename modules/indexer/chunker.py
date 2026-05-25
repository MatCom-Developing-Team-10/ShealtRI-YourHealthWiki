from __future__ import annotations

import math
import re

from core.models import Document

_VALID_STRATEGIES = {"fixed", "paragraph", "semantic"}

# Sentence boundary pattern: split on '.', '!', '?' followed by whitespace or end
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Default similarity threshold below which a topic change is detected
_DEFAULT_SIMILARITY_THRESHOLD = 0.5


class TextChunker:
    """Splits documents into overlapping chunks before indexing.

    Supports three strategies selectable at construction time:

    - **"fixed"** (default): sliding window of *chunk_size* tokens with *overlap*
      tokens repeated at the boundary.  Fast and predictable; may cut mid-sentence.
    - **"paragraph"**: accumulates whole paragraphs (split on ``\\n\\n``) until the
      token budget is reached, then starts a new chunk.  The last paragraph of the
      previous chunk is repeated as overlap so context is not lost at boundaries.
      Preserves natural semantic units — preferred when the source text has clear
      paragraph structure (e.g. HTML-scraped medical articles).
    - **"semantic"**: splits on detected topic changes using sentence-level cosine
      similarity between consecutive sentence embeddings (sentence-transformers).
      A drop below *similarity_threshold* signals a topic boundary.  Produces the
      most semantically coherent chunks at the cost of requiring the
      ``sentence-transformers`` package and a one-time model load.

    Each chunk becomes an independent Document whose doc_id encodes the original
    document id and the chunk index.  The original doc_id is preserved in chunk
    metadata so downstream components can group or deduplicate by source document.
    """

    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 40,
        strategy: str = "fixed",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Maximum number of tokens per chunk (fixed/paragraph strategies).
            overlap: Tokens/sentences repeated at chunk boundaries for context continuity.
            strategy: One of ``"fixed"``, ``"paragraph"``, or ``"semantic"``.
            model_name: sentence-transformers model used by the ``"semantic"`` strategy.
                Ignored for other strategies.  Defaults to ``"all-MiniLM-L6-v2"``
                (fast, small, multilingual-capable).
            similarity_threshold: Cosine similarity below which a sentence boundary
                is treated as a topic change.  Only used by ``"semantic"`` strategy.
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got {strategy!r}")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._strategy = strategy
        self._model_name = model_name
        self._similarity_threshold = similarity_threshold
        self._encoder = None  # lazy-loaded on first semantic chunk call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Document]:
        """Split *doc* into overlapping chunks using the configured strategy.

        Args:
            doc: Source document to split.

        Returns:
            List of Document objects, one per chunk.  A document whose text is
            shorter than chunk_size is returned as a single-element list.
        """
        if self._strategy == "paragraph":
            return self._chunk_by_paragraph(doc)
        if self._strategy == "semantic":
            return self._chunk_semantic(doc)
        return self._chunk_fixed(doc)

    def total_chunks(self, token_count: int) -> int:
        """Return the number of chunks produced for a document with *token_count* tokens."""
        if token_count == 0:
            return 1
        if token_count <= self._chunk_size:
            return 1
        step = self._chunk_size - self._overlap
        return math.ceil((token_count - self._overlap) / step)

    # ------------------------------------------------------------------
    # Strategy: fixed-size sliding window
    # ------------------------------------------------------------------

    def _chunk_fixed(self, doc: Document) -> list[Document]:
        tokens = doc.text.split()
        if not tokens:
            return [doc]

        step = self._chunk_size - self._overlap
        windows = self._sliding_windows(tokens, self._chunk_size, step)
        return self._build_chunks(doc, [" ".join(w) for w in windows])

    # ------------------------------------------------------------------
    # Strategy: paragraph-aware accumulation
    # ------------------------------------------------------------------

    def _chunk_by_paragraph(self, doc: Document) -> list[Document]:
        """Accumulate whole paragraphs up to chunk_size tokens; overlap via last paragraph."""
        paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        if not paragraphs:
            return self._chunk_fixed(doc)

        groups: list[str] = []
        current_paras: list[str] = []
        current_tokens = 0
        last_para: str | None = None  # overlap carrier

        for para in paragraphs:
            para_tokens = len(para.split())

            # A single paragraph that exceeds the budget is kept as its own chunk.
            if para_tokens >= self._chunk_size:
                if current_paras:
                    groups.append("\n\n".join(current_paras))
                    last_para = current_paras[-1]
                    current_paras = []
                    current_tokens = 0
                groups.append(para)
                last_para = para
                continue

            # Adding this paragraph would exceed the budget — flush current group.
            if current_tokens + para_tokens > self._chunk_size and current_paras:
                groups.append("\n\n".join(current_paras))
                last_para = current_paras[-1]
                # Start next chunk with overlap: repeat the last paragraph.
                current_paras = [last_para] if last_para else []
                current_tokens = len(last_para.split()) if last_para else 0

            current_paras.append(para)
            current_tokens += para_tokens

        if current_paras:
            groups.append("\n\n".join(current_paras))

        if not groups:
            return [doc]

        return self._build_chunks(doc, groups)

    # ------------------------------------------------------------------
    # Strategy: semantic topic-change detection
    # ------------------------------------------------------------------

    def _chunk_semantic(self, doc: Document) -> list[Document]:
        """Split on topic changes detected via cosine similarity between sentence embeddings.

        Algorithm (as described in Conf_9-RAG.pdf):
        1. Split text into sentences.
        2. Embed each sentence with a sentence-transformers model.
        3. Compute cosine similarity between consecutive sentence embeddings.
        4. A drop below ``similarity_threshold`` signals a topic change → open new chunk.
        5. Overlap: repeat the last sentence of the previous chunk at the start of the next.

        Falls back to ``_chunk_fixed`` if fewer than 2 sentences are found.
        """
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(doc.text) if s.strip()]
        if len(sentences) < 2:
            return self._chunk_fixed(doc)

        encoder = self._get_encoder()
        embeddings = encoder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

        groups: list[str] = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self._similarity_threshold:
                # Topic change detected — close current chunk, carry last sentence as overlap
                groups.append(" ".join(current))
                last_sentence = current[-1]
                current = [last_sentence, sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            groups.append(" ".join(current))

        return self._build_chunks(doc, groups) if groups else self._chunk_fixed(doc)

    def _get_encoder(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "The 'semantic' chunking strategy requires the sentence-transformers "
                    "package. Install it with: pip install sentence-transformers"
                ) from exc
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity between two numpy vectors."""
        import numpy as np
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_chunks(self, doc: Document, texts: list[str]) -> list[Document]:
        """Wrap a list of text strings into Document objects with correct metadata."""
        total = len(texts)
        chunks: list[Document] = []
        for idx, text in enumerate(texts):
            meta = dict(doc.metadata or {})
            meta["original_doc_id"] = doc.doc_id
            meta["chunk_index"] = idx
            meta["total_chunks"] = total
            meta["chunk_strategy"] = self._strategy
            chunks.append(
                Document(
                    doc_id=f"{doc.doc_id}__chunk_{idx}",
                    text=text,
                    url=doc.url,
                    metadata=meta,
                )
            )
        return chunks

    @staticmethod
    def _sliding_windows(tokens: list[str], size: int, step: int) -> list[list[str]]:
        windows: list[list[str]] = []
        start = 0
        while start < len(tokens):
            windows.append(tokens[start : start + size])
            if start + size >= len(tokens):
                break
            start += step
        return windows
