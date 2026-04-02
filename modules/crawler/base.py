"""Abstract base class for site-specific scrapers.

Each scraper targets a single medical website and knows how to extract
structured content from its HTML pages. The generic crawler handles
everything else: HTTP fetching, robots.txt, delays, queuing, and persistence.

Scrapers must store all site-specific knowledge (CSS selectors, URL patterns,
language detection logic) inside the subclass. The crawler never reaches
into scraper internals.

Metadata contract
-----------------
The Document returned by scrape() must populate doc.metadata with at least
these keys so that downstream modules (indexer, ranker, UI) can filter and
display results correctly:

    title     (str)  — page title as it appears on the site
    source    (str)  — matches source_name (e.g., "mayo_clinic")
    language  (str)  — ISO 639-1 code: "en" or "es"
    date      (str)  — publication or last-modified date, ISO 8601 if known,
                       empty string if not available
    category  (str)  — broad topic tag (e.g., "disease", "symptom", "drug",
                       "wellness"), site-specific but consistent within a scraper

Additional keys are allowed. Unknown keys are ignored by the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import Document


class BaseScraper(ABC):
    """Contract for site-specific HTML scrapers.

    The generic crawler calls can_handle() to select the right scraper
    for a URL, then calls scrape() to extract a Document. The scraper
    never performs HTTP requests — it only parses HTML already fetched
    by the crawler.

    Subclass checklist:
        1. Set ``domain`` to the hostname (e.g., "www.mayoclinic.org").
        2. Set ``source_name`` to the snake_case source identifier
           (e.g., "mayo_clinic"). This becomes the JSONL filename.
        3. Set ``sitemap_urls`` to the list of XML sitemaps for this source.
           Override ``get_sitemap_urls()`` instead if the list must be built
           at runtime.
        4. Implement ``scrape()`` to parse HTML and return a Document
           with the required metadata keys (see module docstring).
        5. Return None from ``scrape()`` for pages with no useful content
           (index pages, login walls, 404 pages, etc.).
        6. Override ``can_handle()`` if domain-level matching is insufficient
           (e.g., you only handle a specific path prefix).

    Example:
        class MayoClinicScraper(BaseScraper):
            domain = "www.mayoclinic.org"
            source_name = "mayo_clinic"
            sitemap_urls = [
                "https://www.mayoclinic.org/sitemap/condition_consolidated_concepts.xml",
            ]

            def scrape(self, url: str, html: str) -> Document | None:
                soup = BeautifulSoup(html, "lxml")
                ...
    """

    #: Hostname this scraper handles, without scheme (e.g., "www.mayoclinic.org").
    #: Must be set as a class attribute in every subclass.
    domain: str

    #: Snake_case identifier used as the JSONL filename in data/raw/.
    #: Must be set as a class attribute in every subclass.
    source_name: str

    #: XML sitemap URLs the crawler should fetch to discover article URLs.
    #: Override get_sitemap_urls() when the list must be built at runtime.
    sitemap_urls: list[str] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce that subclasses declare domain and source_name."""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "domain", None):
            raise TypeError(
                f"{cls.__name__} must define a non-empty class attribute 'domain'"
            )
        if not getattr(cls, "source_name", None):
            raise TypeError(
                f"{cls.__name__} must define a non-empty class attribute 'source_name'"
            )

    @abstractmethod
    def scrape(self, url: str, html: str) -> Document | None:
        """Extract a Document from the raw HTML of a page.

        Args:
            url: Canonical URL of the page (already fetched by the crawler).
            html: Raw HTML content as a string.

        Returns:
            A populated Document if the page contains useful medical content.
            None if the page should be skipped (index, error, login, etc.).

        Note:
            The returned Document must have metadata keys:
            title, source, language, date, category.
            See module docstring for the full contract.
        """
        ...

    def get_sitemap_urls(self) -> list[str]:
        """Return the sitemap URLs used to seed the crawl for this source.

        The default implementation returns a copy of the ``sitemap_urls``
        class attribute. Override this method when the URL list must be
        constructed at runtime (e.g., paginated sitemaps, date-range feeds).

        Returns:
            List of absolute XML sitemap URLs (urlset or sitemapindex format).
            Return an empty list if this scraper does not use sitemaps.
        """
        return list(self.sitemap_urls)

    def can_handle(self, url: str) -> bool:
        """Check whether this scraper should process a given URL.

        The default implementation checks whether self.domain appears
        anywhere in the URL string. Override for stricter matching
        (e.g., only handle URLs under /diseases-conditions/).

        Args:
            url: URL to evaluate.

        Returns:
            True if this scraper can extract content from the URL.
        """
        return self.domain in url
