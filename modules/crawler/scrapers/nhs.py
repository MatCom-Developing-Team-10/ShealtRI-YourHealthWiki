"""NHS (National Health Service) scraper for medical content acquisition.

NHS (nhs.uk) is the UK's National Health Service, providing authoritative
health information for patients and the public.

Page types handled:
    - Conditions  (e.g., /conditions/diabetes/)       → category "disease"
    - Medicines   (e.g., /medicines/metformin/)        → category "drug"
    - Live Well   (e.g., /live-well/eat-well/)         → category "wellness"
    - Mental health (e.g., /mental-health/conditions/) → category "mental-health"

Content structure (confirmed from live HTML):
    - Title:    <main id="maincontent"> → <h1>
    - Content:  all <p> tags inside <main>
    - Date:     <meta name="article:modified_time" content="...">
    - Language: <html lang="..."> attribute

Sitemap:
    Single sitemap index at https://www.nhs.uk/sitemap.xml.
    can_handle() filters to known medical content paths before any fetch occurs.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from core.models import Document
from modules.crawler.base import BaseScraper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maps URL path segment → category label stored in metadata.
_CATEGORY_MAP: dict[str, str] = {
    "conditions": "disease",
    "medicines": "drug",
    "live-well": "wellness",
    "mental-health": "mental-health",
}

# URL path prefixes that contain actual medical content.
# Any URL NOT starting with one of these is rejected by can_handle().
_CONTENT_PATHS: tuple[str, ...] = (
    "/conditions/",
    "/medicines/",
    "/live-well/",
    "/mental-health/",
)

# Minimum character count for the extracted text to be considered valid.
_MIN_CONTENT_LENGTH: int = 50

# Meta tag names/properties to try when looking for the article date,
# checked in order — first match wins.
_DATE_META_ATTRS: tuple[dict[str, str], ...] = (
    {"name": "article:modified_time"},
    {"property": "article:modified_time"},
    {"name": "article:published_time"},
    {"property": "article:published_time"},
)


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------


class NHSScraper(BaseScraper):
    """Scraper for www.nhs.uk medical content pages.

    Uses the sitemap at ``sitemap_urls`` to discover article URLs.
    The crawler fetches those URLs and passes the raw HTML to ``scrape()``,
    which extracts the article using the NHS page structure.

    Language detection:
        Read from ``<html lang="...">`` attribute. NHS pages are in English
        (``lang="en"``).

    Category detection:
        Inferred from the URL path:
            /conditions/    → "disease"
            /medicines/     → "drug"
            /live-well/     → "wellness"
            /mental-health/ → "mental-health"

    Skip conditions:
        Returns None (page skipped) when:
            - No ``<h1>`` is found inside ``<main>``
            - Extracted text is fewer than 50 characters
    """

    domain = "www.nhs.uk"
    source_name = "nhs"
    sitemap_urls = [
        "https://www.nhs.uk/sitemap.xml",
    ]

    def can_handle(self, url: str) -> bool:
        """Accept only URLs under known medical content paths.

        Rejects service-search, contact, campaign, and other non-clinical
        sections before any HTTP request is made.

        Args:
            url: Absolute URL to evaluate.

        Returns:
            True if the URL path starts with a known content prefix.
        """
        path = urlparse(url).path
        return any(path.startswith(prefix) for prefix in _CONTENT_PATHS)

    def scrape(self, url: str, html: str) -> Document | None:
        """Extract a medical Document from an NHS page.

        Args:
            url: Canonical URL of the page (already fetched by the crawler).
            html: Raw HTML content as a string.

        Returns:
            Document with title, content text, and required metadata keys,
            or None if the page has no extractable medical content.
        """
        soup = BeautifulSoup(html, "lxml")

        language = _extract_language(soup)
        title = _extract_title(soup)

        if not title:
            logger.debug("No title found — skipping %s", url)
            return None

        content = _extract_content(soup)

        if len(content) < _MIN_CONTENT_LENGTH:
            logger.debug(
                "Content too short (%d chars) — skipping %s", len(content), url
            )
            return None

        return Document(
            doc_id="",  # overwritten by GenericCrawler with UUID5
            text=content,
            url=url,
            metadata={
                "title": title,
                "source": self.source_name,
                "language": language,
                "date": _extract_date(soup),
                "category": _infer_category(url),
            },
        )


# ---------------------------------------------------------------------------
# Private extraction helpers
# ---------------------------------------------------------------------------


def _extract_language(soup: BeautifulSoup) -> str:
    """Read the ISO 639-1 language code from the <html> tag.

    Args:
        soup: Parsed page.

    Returns:
        Two-letter language code (e.g. "en"). Defaults to "en".
    """
    html_tag = soup.find("html")
    if html_tag:
        lang = html_tag.get("lang", "en")
        return str(lang).split("-")[0].lower()
    return "en"


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Extract the article title from the <h1> inside <main>.

    Falls back to the first <h1> anywhere on the page if <main> is absent.

    Args:
        soup: Parsed page.

    Returns:
        Title string, or None if no <h1> is found.
    """
    main = soup.find("main")
    container = main if main else soup
    h1 = container.find("h1")
    if not h1:
        h1 = soup.find("h1")
    if not h1:
        return None
    title = h1.get_text(strip=True)
    return title if title else None


def _extract_content(soup: BeautifulSoup) -> str:
    """Extract body text from <p> tags inside <main>.

    Skips empty paragraphs and joins non-empty ones with double newlines
    so that paragraph boundaries are preserved for the indexer.

    Args:
        soup: Parsed page.

    Returns:
        Concatenated article text, or "" if no content is found.
    """
    main = soup.find("main")
    if not main:
        return ""

    paragraphs = [
        p.get_text(strip=True)
        for p in main.find_all("p")
        if p.get_text(strip=True)
    ]
    return "\n\n".join(paragraphs)


def _extract_date(soup: BeautifulSoup) -> str:
    """Extract the article date from meta tags.

    Tries ``article:modified_time`` and ``article:published_time`` in that
    order. Returns the first non-empty value as-is (NHS uses a human-readable
    format, e.g. "3 Jul 2025, 11:31 a.m.").

    Args:
        soup: Parsed page.

    Returns:
        Date string as provided by the site, or "" if not found.
    """
    for attrs in _DATE_META_ATTRS:
        tag = soup.find("meta", attrs=attrs)
        if tag:
            value = tag.get("content", "").strip()
            if value:
                return value
    return ""


def _infer_category(url: str) -> str:
    """Infer a topic category from the URL path.

    Args:
        url: Absolute page URL.

    Returns:
        Category string from ``_CATEGORY_MAP``, or "" if path is unknown.
    """
    path = urlparse(url).path
    for segment, category in _CATEGORY_MAP.items():
        if f"/{segment}/" in path:
            return category
    return ""
