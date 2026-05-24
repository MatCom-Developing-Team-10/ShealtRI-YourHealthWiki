import math
from core.models import Document

_VALID_STRATEGIES = {"fixed", "paragraph"}


class TextChunker:
    """Splits documents into overlapping chunks before indexing.

    Supports two strategies selectable at construction time:

    - **"fixed"** (default): sliding window of *chunk_size* tokens with *overlap*
      tokens repeated at the boundary.  Fast and predictable; may cut mid-sentence.
    - **"paragraph"**: accumulates whole paragraphs (split on ``\\n\\n``) until the
      token budget is reached, then starts a new chunk.  The last paragraph of the
      previous chunk is repeated as overlap so context is not lost at boundaries.
      Preserves natural semantic units — preferred when the source text has clear
      paragraph structure (e.g. HTML-scraped medical articles).

    Each chunk becomes an independent Document whose doc_id encodes the original
    document id and the chunk index.  The original doc_id is preserved in chunk
    metadata so downstream components can group or deduplicate by source document.
    """

    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 40,
        strategy: str = "fixed",
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got {strategy!r}")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._strategy = strategy

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
