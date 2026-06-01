"""demo_crawl.py — ShealtRI demo: run the crawler with verbose live logging.

    python demo_crawl.py                  # crawl MedlinePlus, 10 pages
    python demo_crawl.py --source mayo    # only Mayo Clinic, 10 pages
    python demo_crawl.py --source all     # all 3 sources, 10 pages each
    python demo_crawl.py --max-pages 25   # increase limit

Purpose
-------
This script is designed for a LIVE video demo. It runs a bounded crawl
session (default: 10 pages per source) and prints every step to the
console in verbose detail: robots.txt fetch, sitemap parse, each page
crawled, rate-limiting delays, and final statistics.

The downloaded documents are saved to data/raw/ as JSONL files.
When you next start the system (python cli.py), it will detect these new
files and fold them into the existing index (warm start + incremental).

IMPORTANT for demo
------------------
The system's PRIMARY corpus is 20 medical textbooks (PDF) that are
pre-indexed in data/documents/ and data/chroma/. This script demonstrates
the WEB ACQUISITION pipeline — its output supplements the main corpus
rather than replacing it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Verbose logging setup ──────────────────────────────────────────────────


class _ColorFormatter(logging.Formatter):
    """Adds color and structure to crawler logs for readability on screen."""

    GREY = "\033[90m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED + BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        level = f"{color}{record.levelname:<8}{self.RESET}"
        name = f"{self.GREY}{record.name}{self.RESET}"
        return f"  {level} {name}: {record.getMessage()}"


def _setup_logging(verbose: bool = True) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter())
    root_logger.handlers = [handler]

    # Silence noisy third-party libraries
    for noisy in ("urllib3", "requests", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Source configuration ───────────────────────────────────────────────────

def _build_scrapers(source: str) -> list:
    """Build the requested scraper list."""
    from modules.crawler.scrapers.medlineplus import MedlinePlusScraper
    from modules.crawler.scrapers.mayo_clinic import MayoClinicScraper
    from modules.crawler.scrapers.nhs import NHSScraper

    all_scrapers = {
        "medlineplus": MedlinePlusScraper,
        "mayo": MayoClinicScraper,
        "nhs": NHSScraper,
    }

    if source == "all":
        return [cls() for cls in all_scrapers.values()]

    if source not in all_scrapers:
        print(
            f"  ✗  Unknown source '{source}'. "
            f"Choose from: {', '.join(all_scrapers)} or 'all'",
            file=sys.stderr,
        )
        sys.exit(1)

    return [all_scrapers[source]()]


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShealtRI demo: run the medical web crawler with verbose output",
    )
    parser.add_argument(
        "--source", "-s",
        default="medlineplus",
        choices=["medlineplus", "mayo", "nhs", "all"],
        help="Which source(s) to crawl (default: medlineplus)",
    )
    parser.add_argument(
        "--max-pages", "-n",
        type=int,
        default=10,
        metavar="N",
        help="Maximum pages to crawl per source (default: 10)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        metavar="SECS",
        help="Seconds to wait between requests (default: 1.0)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/raw",
        help="Directory to store JSONL output (default: data/raw)",
    )
    args = parser.parse_args()

    _setup_logging(verbose=True)

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║         ShealtRI — Live Crawler Demo                         ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Source(s)   : {args.source}")
    print(f"  Max pages   : {args.max_pages} per source")
    print(f"  Rate limit  : {args.delay}s between requests")
    print(f"  Output dir  : {args.output}")
    print()
    print("  ─── robots.txt and sitemap discovery ───────────────────────────")
    print()

    from modules.crawler.service import CrawlerService
    from modules.crawler.models import CrawlConfig

    scrapers = _build_scrapers(args.source)
    config = CrawlConfig(
        max_pages=args.max_pages,
        delay_seconds=args.delay,
        output_dir=args.output,
        respect_robots=True,
    )

    service = CrawlerService(scrapers=scrapers, config=config)

    t0 = time.monotonic()
    print("  ─── crawling pages ──────────────────────────────────────────────")
    print()
    try:
        result = service.run()
    except KeyboardInterrupt:
        print("\n  Crawl interrupted by user.")
        sys.exit(0)

    elapsed = time.monotonic() - t0

    print()
    print("  ─── crawl summary ───────────────────────────────────────────────")
    print()
    print(f"  Pages visited   : {result.total_visited}")
    print(f"  Pages OK        : {result.total_successful}")
    print(f"  Docs saved      : {result.documents_saved}")
    print(f"  Pages failed    : {result.total_failed}  (robots.txt / empty / errores HTTP)")
    print(f"  Errors          : {len(result.errors)}")
    print(f"  Success rate    : {result.success_rate:.1%}")
    print(f"  Duration        : {elapsed:.1f}s")
    print()

    if result.documents_saved > 0:
        output_path = Path(args.output)
        jsonl_files = list(output_path.glob("*.jsonl"))
        print(f"  JSONL files in {args.output}:")
        for f in sorted(jsonl_files):
            lines = sum(1 for _ in open(f, encoding="utf-8") if _.strip())
            print(f"    {f.name}  ({lines} documents)")
        print()
        print("  ✓  Documentos guardados. En el próximo arranque del sistema")
        print("     (python cli.py), serán indexados automáticamente vía")
        print("     dynamic folding-in sin re-entrenar el SVD completo.")
    else:
        print("  No se guardaron documentos. Revisa la conexión o el robots.txt.")

    print()


if __name__ == "__main__":
    main()
