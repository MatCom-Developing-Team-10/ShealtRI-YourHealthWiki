"""Web content fetcher — downloads pages found via DuckDuckGo search.

Performs two steps for each query:
    1. Search: use the DuckDuckGo API (no API key required) to get a list of
       relevant URLs and titles.
    2. Fetch: download each page with ``requests`` and extract clean text
       using ``BeautifulSoup``.

The resulting ``Document`` objects have the fetched page text as their body
and can be passed directly into any ``BaseRetriever`` or persisted via a
``DocumentStore`` for caching.

Usage:
    fetcher = WebContentFetcher(max_text_length=8000, request_timeout=10)
    docs = fetcher.fetch("hypertension treatment guidelines", max_results=5)
    for doc in docs:
        print(doc.url, len(doc.text))
"""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from core.models import Document

logger = logging.getLogger(__name__)

# Tags whose content is always noise — remove before text extraction.
_NOISE_TAGS = [
    "script", "style", "noscript", "header", "footer", "nav",
    "aside", "form", "button", "iframe", "svg", "img",
]

# Default user-agent to identify the crawler politely.
_USER_AGENT = (
    "ShealtRI-Crawler/1.0 (Medical information retrieval research project; "
    "educational use only)"
)

# Over-fetch pool: ask DDG for more URLs than the caller wants so we still hit
# the target count after the inevitable per-page losses (JS-only pages, 403s,
# parser failures, snippets shorter than ``min_text_length``).
_FETCH_POOL_SIZE = 15


class WebContentFetcher:
    """Searches DuckDuckGo and fetches page content for a text query.

    Each call to ``fetch()`` returns a list of ``Document`` objects built from
    the pages found by DuckDuckGo. Pages that time out, return HTTP errors, or
    contain too little text are skipped gracefully.

    Attributes:
        request_timeout: Seconds to wait before aborting a page download.
        max_text_length: Maximum number of characters to keep from each page.
            Truncation happens at a word boundary when possible.
        min_text_length: Pages with fewer characters after cleaning are skipped.
        user_agent: HTTP User-Agent header sent with every request.
    """

    def __init__(
        self,
        request_timeout: int = 10,
        max_text_length: int = 8000,
        min_text_length: int = 200,
        user_agent: str = _USER_AGENT,
        max_workers: int = 8,
    ) -> None:
        """Initialize the fetcher.

        Args:
            request_timeout: HTTP request timeout in seconds.
            max_text_length: Max characters to keep per fetched page.
            min_text_length: Minimum characters required to keep a page.
            user_agent: User-Agent header for HTTP requests.
            max_workers: Maximum number of pages to download concurrently.
        """
        self.request_timeout = request_timeout
        self.max_text_length = max_text_length
        self.min_text_length = min_text_length
        self.max_workers = max_workers
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def fetch(self, query: str, max_results: int = 5) -> list[Document]:
        """Search DuckDuckGo and fetch content from the result pages.

        Args:
            query: Search query string (plain text, no special operators needed).
            max_results: Maximum number of pages to download. More results mean
                better recall but slower response time.

        Returns:
            List of ``Document`` objects containing the fetched page text.
            May be shorter than ``max_results`` if some pages fail or are too short.
        """
        from core.models import Document  # local import avoids circular deps at module load

        if not query or not query.strip():
            logger.warning("WebContentFetcher.fetch: empty query, returning []")
            return []

        pool_size = max(_FETCH_POOL_SIZE, max_results)
        search_results = self._search_duckduckgo(query, pool_size)
        if not search_results:
            logger.warning(
                "WebContentFetcher: no DuckDuckGo results for query='%s'", query
            )
            return []

        # Download the result pages concurrently. Network latency dominates, so
        # fetching in parallel cuts total time from the sum of every page fetch
        # down to roughly the slowest single fetch. submit() preserves rank
        # order: the futures list stays in submission order.
        def _safe_result(future) -> "Document | None":
            """Resolve a future, isolating any unexpected per-page failure.

            ``_fetch_page`` already returns None on its own handled errors, but an
            unexpected exception inside a worker (e.g. a parser crash) would
            otherwise re-raise here and abort the whole fetch. Swallowing it lets
            the remaining successfully-fetched pages through.
            """
            try:
                return future.result()
            except Exception:
                logger.warning("WebContentFetcher: a page fetch failed; skipping", exc_info=True)
                return None

        max_workers = min(len(search_results), self.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._fetch_page, url, title, snippet, rank)
                for rank, (url, title, snippet) in enumerate(search_results)
            ]
            documents = [doc for f in futures if (doc := _safe_result(f)) is not None]

        # Trim to the caller's budget, preserving DDG rank order.
        kept = documents[:max_results]
        logger.info(
            "WebContentFetcher: fetched %d/%d pages, returning %d for query='%s'",
            len(documents), len(search_results), len(kept), query,
        )
        return kept

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_duckduckgo(
        self, query: str, max_results: int
    ) -> list[tuple[str, str, str]]:
        """Query DuckDuckGo and return (url, title, snippet) tuples.

        Args:
            query: Search query.
            max_results: Maximum number of results to request.

        Returns:
            List of (url, title, snippet) tuples. Empty list on failure.
        """
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))

            results = [
                (r["href"], r.get("title", ""), r.get("body", ""))
                for r in raw
                if r.get("href")
            ]
            logger.debug(
                "DuckDuckGo returned %d results for query='%s'", len(results), query
            )
            return results

        except ImportError:
            logger.error(
                "ddgs is not installed. "
                "Run: pip install ddgs"
            )
            return []
        except Exception as e:
            logger.warning("DuckDuckGo search failed for query='%s': %s", query, e)
            return []

    def _fetch_page(
        self, url: str, title: str, snippet: str, rank: int
    ) -> Document | None:
        """Download a single page and extract its text content.

        Falls back to the DuckDuckGo snippet if the page cannot be downloaded.

        Args:
            url: Page URL to fetch.
            title: Page title from DuckDuckGo (used in metadata).
            snippet: DuckDuckGo result snippet (fallback if page fetch fails).
            rank: Zero-based result rank (used for doc_id generation).

        Returns:
            Document with extracted text, or None if the page is too short.
        """
        from core.models import Document

        text = self._download_text(url)

        if text is None:
            # Fall back to the snippet from DuckDuckGo
            if snippet and len(snippet) >= self.min_text_length:
                text = snippet
                logger.debug("WebContentFetcher: using snippet fallback for %s", url)
            else:
                logger.debug(
                    "WebContentFetcher: skipping %s (fetch failed, snippet too short)",
                    url,
                )
                return None

        if len(text) < self.min_text_length:
            logger.debug(
                "WebContentFetcher: skipping %s (text too short: %d chars)", url, len(text)
            )
            return None

        # Truncate at word boundary
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length].rsplit(" ", 1)[0]

        doc_id = self._make_doc_id(url)
        return Document(
            doc_id=doc_id,
            text=text,
            url=url,
            metadata={
                "title": title,
                "source": "web_search",
                "rank": rank,
            },
        )

    def _download_text(self, url: str) -> str | None:
        """Download a URL and extract readable text from its HTML.

        Args:
            url: Page URL.

        Returns:
            Cleaned page text, or None on network/parsing error.
        """
        try:
            response = self._session.get(
                url, timeout=self.request_timeout, allow_redirects=True
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                logger.debug(
                    "WebContentFetcher: skipping non-HTML response at %s (%s)",
                    url, content_type,
                )
                return None

            return self._extract_text(response.text)

        except requests.exceptions.Timeout:
            logger.debug("WebContentFetcher: timeout fetching %s", url)
            return None
        except requests.exceptions.RequestException as e:
            logger.debug("WebContentFetcher: request error for %s: %s", url, e)
            return None

    def _extract_text(self, html: str) -> str:
        """Parse HTML and extract clean readable text.

        Removes script/style/nav tags, collapses whitespace, and
        returns the body text.

        Args:
            html: Raw HTML string.

        Returns:
            Cleaned plain text.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove noise tags
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        # Prefer article or main content, fall back to body
        content = soup.find("article") or soup.find("main") or soup.body
        if content is None:
            return ""

        # Get text and collapse whitespace
        raw_text = content.get_text(separator=" ")
        text = re.sub(r"\s+", " ", raw_text).strip()
        return text

    @staticmethod
    def _make_doc_id(url: str) -> str:
        """Generate a stable, filesystem-safe document ID from a URL.

        Args:
            url: Source URL.

        Returns:
            Hex-encoded SHA-256 prefix of the URL (16 chars, collision-resistant).
        """
        return "web_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
