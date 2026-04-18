"""Data models exclusive to the crawler module.

Only CrawlConfig and CrawlResult are defined here.
The Document model is imported from core.models — never redefined in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrawlConfig:
    """Configuration for a crawl session.

    Attributes:
        user_agent: HTTP User-Agent header sent with every request.
        delay_seconds: Minimum delay between consecutive requests to the
            same domain. Applies both between sitemap URLs and between
            article pages.
        max_pages: Maximum number of pages to fetch across all sources.
            None means no limit (crawl everything in the queue).
        output_dir: Directory where JSONL files will be written.
        respect_robots: If True, the crawler parses each domain's robots.txt
            and skips any disallowed URL before fetching.
        request_timeout: HTTP read timeout in seconds. Requests exceeding
            this limit are aborted and counted as failures.
    """

    user_agent: str = "MedSRIBot/1.0 (proyecto académico UH)"
    delay_seconds: float = 2.0
    max_pages: int | None = None
    output_dir: str = "data/raw"
    respect_robots: bool = True
    request_timeout: int = 15


@dataclass
class CrawlResult:
    """Summary of a completed crawl session.

    Attributes:
        documents_saved: Number of documents successfully written to disk.
        total_visited: Total URLs attempted (fetched from the network).
        total_successful: URLs that produced a valid Document.
        total_failed: URLs that returned an HTTP error, a network error,
            or whose content the scraper could not parse.
        errors: List of (url, error_message) pairs for every failed URL.
            Useful for debugging and for resuming interrupted crawls.
        duration_seconds: Wall-clock time for the entire crawl session.
    """

    documents_saved: int
    total_visited: int
    total_successful: int
    total_failed: int
    errors: list[tuple[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of visited URLs that produced a valid document.

        Returns:
            Value in [0.0, 1.0]. Returns 0.0 if no URLs were visited.
        """
        if self.total_visited == 0:
            return 0.0
        return self.total_successful / self.total_visited

    def __str__(self) -> str:
        return (
            f"CrawlResult("
            f"saved={self.documents_saved}, "
            f"visited={self.total_visited}, "
            f"ok={self.total_successful}, "
            f"failed={self.total_failed}, "
            f"rate={self.success_rate:.1%}, "
            f"duration={self.duration_seconds:.1f}s)"
        )
