"""scripts/ingest_pdfs.py — Extract text from PDF books and write JSONL for the pipeline.

This is a required pre-processing step before running the system cold start.
Run it once after cloning the repo (or after a demo_reset):

    python scripts/ingest_pdfs.py

What it does
------------
1. Reads every .pdf in data/raw/
2. Extracts text page by page with pypdf (skips blank pages)
3. Writes one JSONL file per PDF into data/raw/
   Format: {"doc_id": "BookName_pN", "text": "...", "url": "", "metadata": {...}}

After this script finishes, run:

    python cli.py --stats

The cold start will read the JSONL files and index the full corpus.

Options
-------
  --min-chars N    Skip pages with fewer than N characters (default: 100)
  --no-overwrite   Skip books whose JSONL already exists (default: overwrite)
  --book FILENAME  Process only this PDF filename (for quick testing)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

# Silence pypdf's "Ignoring wrong pointing object" noise on malformed PDFs
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)

_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _ROOT / "data" / "raw"

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("  %(levelname)-8s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def _book_stem(pdf_path: Path) -> str:
    """Return a clean stem for use in doc IDs and JSONL filenames."""
    return pdf_path.stem


def _jsonl_path(pdf_path: Path) -> Path:
    return _RAW_DIR / f"{_book_stem(pdf_path)}.jsonl"


def extract_pdf(
    pdf_path: Path,
    min_chars: int = 100,
) -> list[dict]:
    """Extract pages from a PDF and return a list of document dicts.

    Args:
        pdf_path: Path to the PDF file.
        min_chars: Pages with fewer characters than this are skipped.

    Returns:
        List of dicts suitable for JSONL serialization.
    """
    try:
        import pypdf  # noqa: PLC0415
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

    reader = pypdf.PdfReader(str(pdf_path))
    book_name = _book_stem(pdf_path)
    docs: list[dict] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""

        text = text.strip()
        if len(text) < min_chars:
            continue

        docs.append({
            "doc_id": f"{book_name}_p{page_num}",
            "text": text,
            "url": "",
            "metadata": {
                "title": book_name,
                "source": "pdf",
                "page": page_num,
                "total_pages": len(reader.pages),
            },
        })

    return docs


def write_jsonl(docs: list[dict], output_path: Path) -> None:
    """Write documents to a JSONL file (one JSON object per line)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Extract PDF books to JSONL for the ShealtRI pipeline",
    )
    parser.add_argument(
        "--min-chars", type=int, default=100, metavar="N",
        help="Skip pages with fewer than N characters (default: 100)",
    )
    parser.add_argument(
        "--no-overwrite", action="store_true",
        help="Skip PDFs whose JSONL already exists",
    )
    parser.add_argument(
        "--book", metavar="FILENAME",
        help="Process only this PDF filename (e.g. 'Harrison_Manual.pdf')",
    )
    args = parser.parse_args()

    if not _RAW_DIR.exists():
        logger.error("data/raw/ not found. Run from the project root.")
        sys.exit(1)

    if args.book:
        pdfs = [_RAW_DIR / args.book]
        if not pdfs[0].exists():
            logger.error("File not found: %s", pdfs[0])
            sys.exit(1)
    else:
        pdfs = sorted(_RAW_DIR.glob("*.pdf"))

    if not pdfs:
        logger.error("No PDF files found in data/raw/")
        sys.exit(1)

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║         ShealtRI — PDF Ingestion                             ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Source dir  : {_RAW_DIR}")
    print(f"  PDFs found  : {len(pdfs)}")
    print(f"  Min chars   : {args.min_chars}")
    print()

    total_pages = 0
    total_skipped = 0
    t_global = time.monotonic()

    for i, pdf_path in enumerate(pdfs, start=1):
        jsonl_out = _jsonl_path(pdf_path)

        if args.no_overwrite and jsonl_out.exists():
            logger.info("[%d/%d] SKIP  %s  (JSONL already exists)", i, len(pdfs), pdf_path.name)
            continue

        logger.info("[%d/%d] Reading  %s", i, len(pdfs), pdf_path.name)
        t0 = time.monotonic()

        docs = extract_pdf(pdf_path, min_chars=args.min_chars)
        skipped = 0

        # Count skipped pages (total pages - extracted)
        try:
            import pypdf  # noqa: PLC0415
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = pypdf.PdfReader(str(pdf_path))
                skipped = len(reader.pages) - len(docs)
        except Exception:  # noqa: BLE001
            pass

        write_jsonl(docs, jsonl_out)

        elapsed = time.monotonic() - t0
        total_pages += len(docs)
        total_skipped += skipped
        logger.info(
            "         ✓  %d pages extracted, %d skipped  (%.1fs)  → %s",
            len(docs), skipped, elapsed, jsonl_out.name,
        )

    elapsed_total = time.monotonic() - t_global
    print()
    print("  ══════════════════════════════════════════════════════════════")
    print(f"  ✓  Ingestion complete.")
    print(f"  Total pages extracted : {total_pages:,}")
    print(f"  Total pages skipped   : {total_skipped:,}  (blank / < {args.min_chars} chars)")
    print(f"  Total time            : {elapsed_total:.1f}s")
    print()
    print("  JSONL files written to data/raw/. Now run:")
    print()
    print("      python cli.py --stats")
    print()
    print("  to index the corpus and train the LSI model.")
    print()


if __name__ == "__main__":
    main()
