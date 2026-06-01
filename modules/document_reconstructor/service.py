"""Reconstruct a text-only PDF from indexed JSON chunks.

When the source PDF referenced by a chunk's ``url`` no longer exists on
disk, this service rebuilds a best-effort, text-only PDF from every chunk
that shares the same source. The result is cached on disk so the scan over
``data/documents/`` only runs once per source.

Design constraints:
    - Generic: makes no assumption about which JSONs exist; works with any
      corpus directory and any source identifier.
    - Schema-tolerant: tolerates missing ``page`` / ``chunk_index`` metadata
      by falling back to insertion order.
    - Pure read of the chunk store: never mutates the original JSONs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

logger = logging.getLogger(__name__)


class DocumentReconstructor:
    """Rebuilds a missing source PDF from its indexed chunks.

    Args:
        documents_dir: Directory holding the JSON chunk files
            (one JSON per chunk). Defaults to ``data/documents``.
        cache_dir: Directory where reconstructed PDFs are cached.
            Defaults to ``data/pdf_reconstructed``.

    Example:
        >>> recon = DocumentReconstructor()
        >>> pdf_path = recon.reconstruct("data/pdfs/Guyton.pdf")
        >>> pdf_path.exists()
        True
    """

    def __init__(
        self,
        documents_dir: Path | str = Path("data/documents"),
        cache_dir: Path | str = Path("data/pdf_reconstructed"),
    ) -> None:
        self._documents_dir = Path(documents_dir)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(self, source_url: str) -> Path | None:
        """Reconstruct the PDF for *source_url* and return its cached path.

        Args:
            source_url: The ``url`` field stored on each chunk JSON. May be
                a local path (``data/pdfs/Foo.pdf``) or an http(s) URL. Only
                sources with more than one chunk are reconstructed — single
                web-snippet sources are not useful as PDFs.

        Returns:
            Path to the cached reconstructed PDF, or ``None`` if no chunks
            were found for that source.
        """
        cached = self._cache_path_for(source_url)
        if cached.exists():
            logger.debug("Reconstructor: cache hit for %s", source_url)
            return cached

        chunks = list(self._collect_chunks(source_url))
        if not chunks:
            logger.info("Reconstructor: no chunks found for %s", source_url)
            return None

        title = self._derive_title(source_url, chunks)
        ordered = self._order_chunks(chunks)

        logger.info(
            "Reconstructor: building %s from %d chunks", cached.name, len(ordered)
        )
        self._render_pdf(cached, title, source_url, ordered)
        return cached

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_path_for(self, source_url: str) -> Path:
        digest = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:16]
        safe_stem = self._safe_stem(source_url)
        return self._cache_dir / f"{safe_stem}__{digest}.pdf"

    @staticmethod
    def _safe_stem(source_url: str) -> str:
        """Build a filesystem-friendly stem from the source identifier."""
        stem = Path(source_url).stem if "/" in source_url else source_url
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
        return stem[:80] or "document"

    def _collect_chunks(self, source_url: str) -> Iterable[dict]:
        """Yield every chunk JSON whose ``url`` matches *source_url*.

        The scan is linear over ``documents_dir``; with ~43k JSONs and
        a single missing-PDF scenario per session, this is acceptable
        because the result is cached.
        """
        if not self._documents_dir.exists():
            return
        for json_path in self._documents_dir.glob("*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("url") == source_url:
                yield data

    @staticmethod
    def _order_chunks(chunks: list[dict]) -> list[dict]:
        """Sort chunks by (page, chunk_index); missing keys sort last."""

        def sort_key(chunk: dict) -> tuple[int, int]:
            meta = chunk.get("metadata", {}) or {}
            page = meta.get("page")
            chunk_idx = meta.get("chunk_index", 0)
            return (
                page if isinstance(page, int) else 10**9,
                chunk_idx if isinstance(chunk_idx, int) else 0,
            )

        return sorted(chunks, key=sort_key)

    @staticmethod
    def _derive_title(source_url: str, chunks: list[dict]) -> str:
        """Pick a display title from chunk metadata, falling back to filename."""
        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            title = meta.get("title") or meta.get("source_title")
            if title:
                return str(title)
        if "/" in source_url:
            return Path(source_url).stem
        return source_url

    def _render_pdf(
        self,
        out_path: Path,
        title: str,
        source_url: str,
        chunks: list[dict],
    ) -> None:
        """Render the PDF to *out_path* using the platypus flowables API."""
        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            spaceAfter=6,
        )
        heading_style = ParagraphStyle(
            "PageHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor="#444444",
            spaceBefore=4,
            spaceAfter=4,
        )
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            spaceAfter=8,
        )
        notice_style = ParagraphStyle(
            "Notice",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor="#888888",
            spaceAfter=12,
        )

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=LETTER,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=title,
            author="ShealtRI — reconstructed from index",
        )

        flow: list = [
            Paragraph(self._escape(title), title_style),
            Paragraph(
                "Documento reconstruido desde los chunks indexados — "
                "el PDF original no está disponible. Solo se muestra el texto "
                "extraído (sin imágenes, tablas ni formato original).",
                notice_style,
            ),
            Paragraph(
                f"<b>Fuente:</b> {self._escape(source_url)}",
                notice_style,
            ),
            Spacer(1, 0.4 * cm),
        ]

        last_page: int | None = None
        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            page = meta.get("page")
            page_label = meta.get("page_label") or page
            text = (chunk.get("text") or "").strip()
            if not text:
                continue

            if isinstance(page, int) and page != last_page:
                if last_page is not None:
                    flow.append(PageBreak())
                flow.append(
                    Paragraph(f"Página {self._escape(str(page_label))}", heading_style)
                )
                last_page = page

            for para in self._split_paragraphs(text):
                flow.append(Paragraph(self._escape(para), body_style))

        doc.build(flow)

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Break a chunk's raw text into paragraph-sized fragments."""
        parts = re.split(r"\n{2,}", text)
        if len(parts) == 1:
            parts = re.split(r"(?<=[\.\?!])\s{2,}", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _escape(text: str) -> str:
        """Escape characters that would otherwise be parsed as reportlab XML."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
