"""Scraper registry: maps URLs to the correct site-specific scraper.

The registry is populated once at startup and consulted by GenericCrawler
for every URL it is about to process.
"""

from __future__ import annotations

import logging

from modules.crawler.base import BaseScraper

logger = logging.getLogger(__name__)


class ScraperRegistry:
    """Maps a URL to the first registered scraper that can handle it.

    Scrapers are evaluated in registration order. The first scraper whose
    ``can_handle(url)`` returns True is selected. This means registration
    order matters when scrapers have overlapping domains.

    Example:
        registry = ScraperRegistry()
        registry.register(MayoClinicScraper())
        registry.register(MedlinePlusScraper())

        scraper = registry.get("https://www.mayoclinic.org/diseases-conditions/...")
        # → MayoClinicScraper instance
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._scrapers: list[BaseScraper] = []

    def register(self, scraper: BaseScraper) -> None:
        """Add a scraper to the registry.

        Args:
            scraper: Instantiated scraper to register.

        Raises:
            TypeError: If scraper is not a BaseScraper instance.
        """
        if not isinstance(scraper, BaseScraper):
            raise TypeError(
                f"Expected a BaseScraper instance, got {type(scraper).__name__}"
            )
        self._scrapers.append(scraper)
        logger.debug(
            "Registered scraper '%s' for domain '%s'",
            type(scraper).__name__,
            scraper.domain,
        )

    def get(self, url: str) -> BaseScraper | None:
        """Return the first registered scraper that can handle url.

        Iterates in registration order and calls ``scraper.can_handle(url)``.

        Args:
            url: Absolute URL to match.

        Returns:
            First matching scraper, or None if no scraper matches.
        """
        for scraper in self._scrapers:
            if scraper.can_handle(url):
                return scraper
        return None

    def all_scrapers(self) -> list[BaseScraper]:
        """Return a snapshot of all registered scrapers in registration order.

        Used by GenericCrawler to collect sitemap URLs at startup.

        Returns:
            List of all registered scrapers.
        """
        return list(self._scrapers)

    def __len__(self) -> int:
        """Return the number of registered scrapers."""
        return len(self._scrapers)

    def __repr__(self) -> str:
        names = [type(s).__name__ for s in self._scrapers]
        return f"ScraperRegistry({names})"
