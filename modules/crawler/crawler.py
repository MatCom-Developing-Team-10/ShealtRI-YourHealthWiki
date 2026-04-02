"""Generic crawler for medical website data acquisition.

The crawler owns all I/O: HTTP sessions, robots.txt enforcement, delay
scheduling, sitemap parsing, and persistence. It contains no site-specific
HTML parsing logic — that responsibility belongs entirely to BaseScraper
subclasses.

Crawl strategy
--------------
URL discovery is sitemap-only. The crawler fetches each scraper's declared
sitemaps, expands them (including sitemapindex → urlset recursion), and
builds a flat queue of (url, scraper) pairs. Link-following from crawled
pages is intentionally NOT implemented; sitemaps are authoritative for all
three target sources.

Deduplication
-------------
A ``visited_urls`` set prevents re-fetching URLs that appear in multiple
sitemaps or in sitemap indexes that reference overlapping urlsets.

doc_id assignment
-----------------
The crawler always overwrites the doc_id on returned Documents with a
UUID5 derived from the URL. This makes IDs stable across re-crawls and
independent of what each scraper chose to put in the field.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.models import Document
from infra.storage import RawDocumentStorage, RawStorageError
from modules.crawler.base import BaseScraper
from modules.crawler.models import CrawlConfig, CrawlResult
from modules.crawler.registry import ScraperRegistry

logger = logging.getLogger(__name__)

# Flush accumulated documents to storage every N docs per source.
_BATCH_FLUSH_SIZE: int = 50

# Maximum recursion depth when expanding sitemapindex files.
_MAX_SITEMAP_DEPTH: int = 3


class GenericCrawler:
    """HTTP-first crawler that delegates HTML parsing to site-specific scrapers.

    The crawler owns all I/O: HTTP sessions, robots.txt enforcement, delay
    scheduling, sitemap parsing, and persistence. It never contains
    site-specific HTML parsing logic.

    Lifecycle:
        1. ``__init__``: receive scrapers, config, storage; build registry.
        2. ``crawl()``: seed URL queue from sitemaps, drain the queue
           page by page, store results incrementally.

    Thread-safety:
        Not thread-safe. ``crawl()`` is a single-threaded sequential loop.
        Concurrent crawling would complicate robots.txt enforcement and
        rate-limiting without meaningful gain for an academic corpus.

    Example:
        config = CrawlConfig(max_pages=500)
        crawler = GenericCrawler(
            scrapers=[MayoClinicScraper()],
            config=config,
            storage=RawDocumentStorage("data/raw"),
        )
        result = crawler.crawl()
        print(result)
    """

    def __init__(
        self,
        scrapers: list[BaseScraper],
        config: CrawlConfig,
        storage: RawDocumentStorage,
    ) -> None:
        """Initialize the crawler.

        Args:
            scrapers: Site-specific scraper instances. Each must have
                ``domain``, ``source_name``, and ``get_sitemap_urls()`` set.
            config: Controls delays, page limits, robots.txt, timeouts.
            storage: Where crawled documents are persisted as JSONL.
        """
        self._config = config
        self._storage = storage
        self._registry = ScraperRegistry()
        for scraper in scrapers:
            self._registry.register(scraper)

        # Per-domain robots.txt cache: domain → RobotFileParser
        self._robots_cache: dict[str, RobotFileParser] = {}

        # Per-domain last-request timestamp for delay enforcement
        self._last_request_time: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> CrawlResult:
        """Execute the full crawl and return a summary.

        Steps:
            1. Build an HTTP session with retry policy and User-Agent.
            2. For every registered scraper, fetch its sitemaps and expand
               them into a queue of (url, scraper) pairs.
            3. Drain the queue: check robots.txt, enforce delay, fetch HTML,
               call scraper.scrape(), persist Documents, update counters.
            4. Flush any remaining per-source document batches.
            5. Return a CrawlResult with final statistics.

        Returns:
            CrawlResult summarising the session.

        Note:
            No individual failure aborts the crawl. HTTP errors, parse
            errors, and scraper exceptions are logged and counted as
            failures. Only storage I/O errors (disk full, permissions) can
            interrupt the loop, and even then only for the current batch.
        """
        start_time = time.monotonic()

        # Counters
        total_visited: int = 0
        total_successful: int = 0
        total_failed: int = 0
        documents_saved: int = 0
        errors: list[tuple[str, str]] = []

        # Per-source document accumulator: source_name → [Document, ...]
        batches: dict[str, list[Document]] = {}

        session = self._setup_session()

        # --- Phase 1: build the crawl queue from sitemaps ---
        queue: deque[tuple[str, BaseScraper]] = deque()
        visited_urls: set[str] = set()

        for scraper in self._registry.all_scrapers():
            sitemap_urls = scraper.get_sitemap_urls()
            if not sitemap_urls:
                logger.warning(
                    "Scraper '%s' has no sitemap URLs — nothing to crawl",
                    type(scraper).__name__,
                )
                continue

            for sitemap_url in sitemap_urls:
                pairs = self._collect_urls_from_sitemap(
                    sitemap_url, scraper, session
                )
                for url, s in pairs:
                    if url not in visited_urls:
                        visited_urls.add(url)
                        queue.append((url, s))

            logger.info(
                "Scraper '%s': %d unique URLs queued from %d sitemap(s)",
                type(scraper).__name__,
                sum(1 for _, s in queue if s is scraper),
                len(sitemap_urls),
            )

        logger.info("Total URLs in crawl queue: %d", len(queue))

        # --- Phase 2: drain the queue ---
        while queue:
            if self._config.max_pages is not None and total_visited >= self._config.max_pages:
                logger.info(
                    "Reached max_pages limit (%d). Stopping crawl.",
                    self._config.max_pages,
                )
                break

            url, scraper = queue.popleft()
            domain = urlparse(url).netloc

            # robots.txt check
            if self._config.respect_robots:
                if not self._is_allowed_by_robots(url, session):
                    logger.debug("robots.txt: disallowed — skipping %s", url)
                    continue

            # Delay
            self._wait_for_domain(domain)

            # Fetch
            html = self._fetch(url, session)
            total_visited += 1

            if html is None:
                total_failed += 1
                errors.append((url, "HTTP fetch failed"))
                continue

            # Scrape
            try:
                document = scraper.scrape(url, html)
            except Exception as exc:
                logger.warning("Scraper error on %s: %s", url, exc)
                total_failed += 1
                errors.append((url, f"Scraper exception: {exc}"))
                continue

            if document is None:
                # Valid skip — page had no useful content
                logger.debug("Scraper returned None for %s — skipped", url)
                total_failed += 1
                continue

            # Assign stable doc_id
            document = Document(
                doc_id=self._generate_doc_id(url),
                text=document.text,
                url=document.url or url,
                metadata=document.metadata,
            )

            total_successful += 1

            # Accumulate in batch
            source = scraper.source_name
            batches.setdefault(source, []).append(document)

            # Incremental flush
            if len(batches[source]) >= _BATCH_FLUSH_SIZE:
                saved = self._flush_batch(batches[source], source)
                documents_saved += saved
                batches[source] = []

        # --- Phase 3: final flush ---
        for source, docs in batches.items():
            if docs:
                saved = self._flush_batch(docs, source)
                documents_saved += saved

        duration = time.monotonic() - start_time

        result = CrawlResult(
            documents_saved=documents_saved,
            total_visited=total_visited,
            total_successful=total_successful,
            total_failed=total_failed,
            errors=errors,
            duration_seconds=duration,
        )
        logger.info("Crawl complete. %s", result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_session(self) -> requests.Session:
        """Create and configure the HTTP session with retry policy.

        Retries up to 3 times on transient server errors (429, 500, 502,
        503, 504) with exponential backoff. All other errors are handled
        at the call site.

        Returns:
            Configured requests.Session ready for use.
        """
        session = requests.Session()
        session.headers.update({"User-Agent": self._config.user_agent})

        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _collect_urls_from_sitemap(
        self,
        sitemap_url: str,
        scraper: BaseScraper,
        session: requests.Session,
        depth: int = 0,
    ) -> list[tuple[str, BaseScraper]]:
        """Fetch and parse a sitemap, returning (url, scraper) pairs.

        Handles both sitemap formats:
            - ``<urlset>``: leaf sitemap; extract all ``<loc>`` elements.
            - ``<sitemapindex>``: index sitemap; recursively fetch each
              child sitemap up to ``_MAX_SITEMAP_DEPTH``.

        Only URLs for which ``scraper.can_handle(url)`` returns True are
        included. robots.txt is NOT checked for sitemap URLs themselves.

        Args:
            sitemap_url: Absolute URL of the sitemap XML.
            scraper: Scraper instance associated with this sitemap.
            session: Active HTTP session.
            depth: Current recursion depth (caller passes 0).

        Returns:
            List of (article_url, scraper) pairs ready to enqueue.
            Returns an empty list on any fetch or parse error.
        """
        if depth > _MAX_SITEMAP_DEPTH:
            logger.warning(
                "Sitemap recursion depth exceeded at %s — stopping", sitemap_url
            )
            return []

        try:
            response = session.get(
                sitemap_url, timeout=self._config.request_timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch sitemap %s: %s", sitemap_url, exc)
            return []

        try:
            soup = BeautifulSoup(response.text, "xml")
        except Exception as exc:
            logger.warning("Failed to parse sitemap %s: %s", sitemap_url, exc)
            return []

        # sitemapindex → recurse into child sitemaps
        if soup.find("sitemapindex"):
            results: list[tuple[str, BaseScraper]] = []
            child_locs = soup.find_all("sitemap")
            for child in child_locs:
                loc_tag = child.find("loc")
                if loc_tag and loc_tag.text:
                    child_url = loc_tag.text.strip()
                    results.extend(
                        self._collect_urls_from_sitemap(
                            child_url, scraper, session, depth + 1
                        )
                    )
            return results

        # urlset → extract article URLs
        pairs: list[tuple[str, BaseScraper]] = []
        url_tags = soup.find_all("url")
        for tag in url_tags:
            loc_tag = tag.find("loc")
            if not loc_tag or not loc_tag.text:
                continue
            article_url = loc_tag.text.strip()
            if scraper.can_handle(article_url):
                pairs.append((article_url, scraper))

        logger.debug(
            "Sitemap %s → %d URLs (depth=%d)", sitemap_url, len(pairs), depth
        )
        return pairs

    def _is_allowed_by_robots(
        self, url: str, session: requests.Session
    ) -> bool:
        """Check whether url is allowed by the domain's robots.txt.

        Fetches robots.txt once per domain and caches the result.
        Fails open: if robots.txt cannot be fetched or parsed, the URL
        is assumed to be allowed.

        Args:
            url: URL to check.
            session: Active HTTP session (used only on cache miss).

        Returns:
            True if the URL may be fetched, False if disallowed.
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        scheme = parsed.scheme

        if domain not in self._robots_cache:
            robots_url = f"{scheme}://{domain}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                response = session.get(
                    robots_url, timeout=self._config.request_timeout
                )
                parser.parse(response.text.splitlines())
            except Exception as exc:
                logger.warning(
                    "Could not fetch robots.txt for %s: %s — assuming allowed",
                    domain,
                    exc,
                )
            self._robots_cache[domain] = parser

        return self._robots_cache[domain].can_fetch(
            self._config.user_agent, url
        )

    def _fetch(self, url: str, session: requests.Session) -> str | None:
        """Perform an HTTP GET and return the response body as a string.

        Args:
            url: Absolute URL to fetch.
            session: Active HTTP session.

        Returns:
            HTML string on HTTP 200, None on any error.
        """
        try:
            response = session.get(url, timeout=self._config.request_timeout)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
        except requests.Timeout:
            logger.warning("Timeout fetching %s", url)
        except requests.ConnectionError as exc:
            logger.warning("Connection error fetching %s: %s", url, exc)
        except requests.RequestException as exc:
            logger.warning("Request failed for %s: %s", url, exc)
        return None

    def _wait_for_domain(self, domain: str) -> None:
        """Enforce the configured inter-request delay for a domain.

        Tracks the timestamp of the last request per domain. Sleeps for
        the remaining time if the delay has not yet elapsed.

        Args:
            domain: Hostname (e.g., "www.mayoclinic.org").
        """
        now = time.monotonic()
        last = self._last_request_time.get(domain)
        if last is not None:
            elapsed = now - last
            wait = self._config.delay_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_time[domain] = time.monotonic()

    def _flush_batch(self, docs: list[Document], source_name: str) -> int:
        """Persist a batch of documents, returning the count saved.

        Args:
            docs: Documents to persist.
            source_name: JSONL file identifier.

        Returns:
            Number of documents successfully written.
        """
        try:
            return self._storage.save_batch(docs, source_name)
        except RawStorageError as exc:
            logger.error(
                "Storage failure for source '%s': %s — %d documents lost",
                source_name,
                exc,
                len(docs),
            )
            return 0

    @staticmethod
    def _generate_doc_id(url: str) -> str:
        """Generate a stable, unique document ID from a URL.

        Uses UUID5 with the URL as the name. Re-crawling the same URL
        produces the same doc_id, enabling deduplication downstream.

        Args:
            url: Canonical URL of the page.

        Returns:
            UUID5 string (lowercase, hyphenated).
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, url))
