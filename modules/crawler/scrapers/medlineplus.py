"""MedlinePlus scraper for medical content acquisition.

MedlinePlus (medlineplus.gov) is a U.S. National Library of Medicine service
providing authoritative health information for patients and consumers.

Page types handled:
    - Health topic pages  (e.g., /diabetes.html)           → category "health-topic"
    - Spanish health topics (e.g., /spanish/diabetes.html) → category "health-topic", language "es"
    - Medical encyclopedia  (e.g., /ency/article/000305)   → category "reference"

Content structure (confirmed from live HTML):
    - Title:   <article> → <h1 class="with-also">
    - Summary: <section id="topsum_section"> → all <p> tags (the actual medical summary)
    - Date:    <meta name="DC.Date.Modified" content="YYYY-MM-DD">
    - Language: <html lang="..."> attribute

Sitemap:
    Single urlset at https://medlineplus.gov/sitemap.xml (~1000+ URLs).
    The sitemap mixes medical content with index/utility pages; can_handle()
    filters to content-only URLs before any fetch occurs.
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

# Minimum character count for extracted content to be considered valid.
_MIN_CONTENT_LENGTH: int = 50


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------


class MedlinePlusScraper(BaseScraper):
    """Scraper for medlineplus.gov health topic and encyclopedia pages.

    URL filtering:
        - Root-level ``/[topic].html`` pages (English health topics)
        - ``/spanish/[topic].html`` pages (Spanish health topics)
        - ``/ency/article/...`` pages (medical encyclopedia)
        - All other paths are rejected by can_handle().

    Content extraction:
        Title comes from the ``<h1>`` inside the ``<article>`` tag.
        Body text comes exclusively from ``<section id="topsum_section">``,
        which contains the authoritative medical summary paragraphs.
        The other sections on the page are curated link lists, not prose.

    Language:
        Read from the ``<html lang="...">`` attribute. Spanish topic pages
        at ``/spanish/`` also have ``lang="es"`` set by MedlinePlus.

    Date:
        Read from ``<meta name="DC.Date.Modified" content="YYYY-MM-DD">``,
        which MedlinePlus consistently provides on all content pages.
    """

    domain = "medlineplus.gov"
    source_name = "medlineplus"
    sitemap_urls = [
        "https://medlineplus.gov/sitemap.xml",
    ]

    def can_handle(self, url: str) -> bool:
        """Accept only medical content URLs from MedlinePlus.

        Accepted patterns:
            /[topic].html               — root-level English health topics
            /spanish/[topic].html       — Spanish health topics
            /ency/article/[id]          — medical encyclopedia articles

        Rejected patterns (index pages, utilities, feeds, organizations):
            /organizations/...
            /druginfo/...
            /all_*.html
            /connect/
            /*/feeds/
            etc.

        Args:
            url: Absolute URL to evaluate.

        Returns:
            True if the URL points to a medical content page.
        """
        path = urlparse(url).path

        # Root-level health topics: /diabetes.html, /heartdisease.html, etc.
        # These have exactly one slash and end in .html.
        if path.endswith(".html") and path.count("/") == 1:
            return True

        # Spanish health topics: /spanish/diabetes.html
        if path.startswith("/spanish/") and path.endswith(".html"):
            return True

        # Medical encyclopedia articles: /ency/article/000305.htm
        if path.startswith("/ency/article/"):
            return True

        return False

    def scrape(self, url: str, html: str) -> Document | None:
        """Extract a medical Document from a MedlinePlus page.

        Args:
            url: Canonical URL of the page (already fetched by the crawler).
            html: Raw HTML content as a string.

        Returns:
            Document with title, summary text, and required metadata keys,
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
        Two-letter language code ("en" or "es"). Defaults to "en".
    """
    html_tag = soup.find("html")
    if html_tag:
        lang = html_tag.get("lang", "en")
        return str(lang).split("-")[0].lower()
    return "en"


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Extract the article title from <h1> inside <article>.

    Falls back to the first <h1> on the page if <article> is absent.

    Args:
        soup: Parsed page.

    Returns:
        Title string, or None if no <h1> is found.
    """
    article = soup.find("article")
    container = article if article else soup
    h1 = container.find("h1")
    if not h1:
        h1 = soup.find("h1")
    if not h1:
        return None
    title = h1.get_text(strip=True)
    return title if title else None


def _extract_content(soup: BeautifulSoup) -> str:
    """Extract the medical summary from <section id="topsum_section">.

    MedlinePlus health topic pages have a dedicated summary section with
    authoritative medical paragraphs. The other sections on the page are
    curated link lists (not prose), so they are intentionally excluded.

    Falls back to all <p> tags inside <article> if the summary section
    is absent (e.g., encyclopedia pages use a different layout).

    Args:
        soup: Parsed page.

    Returns:
        Concatenated summary text, or "" if no content is found.
    """
    # Primary: health topic summary section
    topsum = soup.find("section", id="topsum_section")
    if topsum:
        paragraphs = [
            p.get_text(strip=True)
            for p in topsum.find_all("p")
            if p.get_text(strip=True)
        ]
        if paragraphs:
            return "\n\n".join(paragraphs)

    # Fallback: all paragraphs inside <article> (encyclopedia pages)
    article = soup.find("article")
    if article:
        paragraphs = [
            p.get_text(strip=True)
            for p in article.find_all("p")
            if p.get_text(strip=True)
        ]
        return "\n\n".join(paragraphs)

    return ""


def _extract_date(soup: BeautifulSoup) -> str:
    """Extract the last-modified date from Dublin Core meta tags.

    MedlinePlus consistently sets <meta name="DC.Date.Modified"> on all
    content pages in YYYY-MM-DD format.

    Args:
        soup: Parsed page.

    Returns:
        Date string (YYYY-MM-DD), or "" if not found.
    """
    tag = soup.find("meta", attrs={"name": "DC.Date.Modified"})
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
        Category string: "health-topic", "reference", or "".
    """
    path = urlparse(url).path

    if path.startswith("/ency/article/"):
        return "reference"

    # Root-level and /spanish/ pages are health topic summaries
    if path.endswith(".html"):
        return "health-topic"

    return ""
