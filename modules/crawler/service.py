"""CrawlerService — single entry point for initiating a crawl session.

Application code (CLI, tests, pipeline) should instantiate CrawlerService
rather than GenericCrawler directly. This facade assembles the three
collaborating objects (GenericCrawler, RawDocumentStorage, CrawlConfig) so
callers do not need to know about the internal wiring.

Usage:
    from modules.crawler import CrawlerService, CrawlConfig
    from modules.crawler.scrapers.mayo_clinic import MayoClinicScraper

    service = CrawlerService(
        scrapers=[MayoClinicScraper()],
        config=CrawlConfig(max_pages=500),
    )
    result = service.run()
    print(result)
"""

from __future__ import annotations

import logging

from infra.storage import RawDocumentStorage
from modules.crawler.base import BaseScraper
from modules.crawler.crawler import GenericCrawler
from modules.crawler.models import CrawlConfig, CrawlResult

logger = logging.getLogger(__name__)


class CrawlerService:
    """Facade that assembles and runs a crawl session.

    Combines GenericCrawler, RawDocumentStorage, and CrawlConfig into a
    single cohesive object. Callers provide scrapers and an optional config;
    the service handles the rest.

    Attributes:
        _scrapers: List of site-specific scraper instances to use.
        _config: Crawl configuration (delays, limits, output dir, etc.).
        _storage: Storage backend writing to config.output_dir.

    Example:
        service = CrawlerService(
            scrapers=[MayoClinicScraper(), MedlinePlusScraper()],
            config=CrawlConfig(delay_seconds=1.5, max_pages=1000),
        )
        result = service.run()
        print(f"Saved {result.documents_saved} documents in {result.duration_seconds:.1f}s")
    """

    def __init__(
        self,
        scrapers: list[BaseScraper],
        config: CrawlConfig | None = None,
    ) -> None:
        """Initialize the crawler service.

        Args:
            scrapers: Instantiated scraper objects. Must be non-empty.
            config: Crawl configuration. Uses CrawlConfig defaults if None.

        Raises:
            ValueError: If scrapers list is empty.
        """
        if not scrapers:
            raise ValueError("At least one scraper must be provided.")

        self._scrapers = scrapers
        self._config = config or CrawlConfig()
        self._storage = RawDocumentStorage(self._config.output_dir)

    def run(self) -> CrawlResult:
        """Assemble the crawler and execute the crawl session.

        Returns:
            CrawlResult with statistics for the completed session.
        """
        logger.info(
            "Starting crawl session: %d scraper(s), max_pages=%s, delay=%.1fs",
            len(self._scrapers),
            self._config.max_pages,
            self._config.delay_seconds,
        )
        crawler = GenericCrawler(
            scrapers=self._scrapers,
            config=self._config,
            storage=self._storage,
        )
        return crawler.crawl()
