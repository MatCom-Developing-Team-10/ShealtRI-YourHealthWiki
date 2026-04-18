"""Mayo Clinic scraper for medical content acquisition.

Extracts structured content from Mayo Clinic article pages using the
AEM (Adobe Experience Manager) component structure found in the site.

Supported content sections (via sitemap — sourced from robots.txt):
    English:  Diseases & Conditions, Symptoms, Articles, FAQ, Drugs, Procedures
    Spanish:  Diseases & Conditions, Articles, FAQ, Procedures

URL filtering:
    Only URLs under known medical content paths are processed.
    Appointment, career, and administrative pages are rejected by
    can_handle() before any fetch occurs.

Metadata produced per document:
    title     — article title from the single <h1> in #main-content
    source    — "mayo_clinic"
    language  — ISO 639-1 code read from <html lang="...">
    date      — ISO 8601 date from meta tags, or "" if unavailable
    category  — topic tag inferred from URL path segment
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
    "diseases-conditions": "disease",
    "symptoms": "symptom",
    "tests-procedures": "procedure",
    "drugs-supplements": "drug",
    "healthy-lifestyle": "wellness",
}

# URL path prefixes that contain actual medical content.
# Any URL NOT starting with one of these is rejected by can_handle().
_CONTENT_PATHS: tuple[str, ...] = (
    "/diseases-conditions/",
    "/symptoms/",
    "/tests-procedures/",
    "/drugs-supplements/",
    "/healthy-lifestyle/",
    "/es/diseases-conditions/",
    "/es/symptoms/",
    "/es/tests-procedures/",
    "/es/drugs-supplements/",
    "/es/healthy-lifestyle/",
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
    {"property": "og:updated_time"},
)


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------


class MayoClinicScraper(BaseScraper):
    """Scraper for www.mayoclinic.org medical content pages.

    Uses the ten XML sitemaps declared in ``sitemap_urls`` to discover
    article URLs (6 English + 4 Spanish, as listed in Mayo Clinic's robots.txt). The crawler fetches those URLs and passes the raw HTML
    to ``scrape()``, which extracts the article using AEM component
    selectors specific to the Mayo Clinic site.

    Language detection:
        Read from ``<html lang="...">`` attribute, present on every page.
        English pages have ``lang="en"``; Spanish pages have ``lang="es"``.
        The ``spanish_*`` sitemaps contain Spanish URLs, but we derive the
        language from the tag rather than the sitemap filename for robustness.

    Category detection:
        Inferred from the URL path:
            /diseases-conditions/ → "disease"
            /symptoms/            → "symptom"
            /tests-procedures/    → "procedure"
            /drugs-supplements/   → "drug"
            /healthy-lifestyle/   → "wellness"

    Skip conditions:
        Returns None (page skipped) when:
            - No ``<h1>`` is found inside ``#main-content``
            - Extracted text is fewer than 50 characters
    """

    domain = "www.mayoclinic.org"
    source_name = "mayo_clinic"
    sitemap_urls = [
        # English — sourced from robots.txt
        "https://www.mayoclinic.org/condition_consolidated_concepts.xml",
        "https://www.mayoclinic.org/symptoms_all.xml",
        "https://www.mayoclinic.org/patient_consumer_web.xml",
        "https://www.mayoclinic.org/patient_consumer_faq.xml",
        "https://www.mayoclinic.org/patient_consumer_drug.xml",
        "https://www.mayoclinic.org/procedure_concepts.xml",
        # Spanish — sourced from robots.txt
        "https://www.mayoclinic.org/spanish_condition_consolidated_concepts.xml",
        "https://www.mayoclinic.org/spanish_patient_consumer_web.xml",
        "https://www.mayoclinic.org/spanish_patient_consumer_faq.xml",
        "https://www.mayoclinic.org/spanish_procedure_concepts.xml",
    ]

    def can_handle(self, url: str) -> bool:
        """Accept only URLs under known medical content paths.

        Rejects appointments, careers, about-mayo-clinic, and other
        non-clinical sections before any HTTP request is made.

        Args:
            url: Absolute URL to evaluate.

        Returns:
            True if the URL path starts with a known content prefix.
        """
        path = urlparse(url).path
        return any(path.startswith(prefix) for prefix in _CONTENT_PATHS)

    def scrape(self, url: str, html: str) -> Document | None:
        """Extract a medical Document from a Mayo Clinic article page.

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
        Two-letter language code (e.g. "en", "es"). Defaults to "en".
    """
    html_tag = soup.find("html")
    if html_tag:
        lang = html_tag.get("lang", "en")
        # Normalize "en-US", "es-MX", etc. → "en", "es"
        return str(lang).split("-")[0].lower()
    return "en"


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Extract the article title from the single <h1> on the page.

    Searches inside ``#main-content`` first. If not found there (Mayo Clinic's
    AEM layout places the ``<h1>`` outside the main content container), falls
    back to the first ``<h1>`` anywhere on the page.

    Args:
        soup: Parsed page.

    Returns:
        Title string, or None if no <h1> is found.
    """
    main = soup.find(id="main-content")
    container = main if main else soup
    h1 = container.find("h1")
    if not h1 and main:
        # Mayo Clinic's AEM layout places <h1> outside #main-content
        h1 = soup.find("h1")
    if not h1:
        return None
    title = h1.get_text(strip=True)
    return title if title else None


def _extract_content(soup: BeautifulSoup) -> str:
    """Extract body text from <p> tags inside #main-content.

    Skips empty paragraphs and joins non-empty ones with double newlines
    so that paragraph boundaries are preserved for the indexer.

    Args:
        soup: Parsed page.

    Returns:
        Concatenated article text, or "" if no content is found.
    """
    main = soup.find(id="main-content")
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

    Tries ``article:modified_time``, ``article:published_time``, and
    ``og:updated_time`` in that order. Returns the first non-empty value.

    Args:
        soup: Parsed page.

    Returns:
        Date string (ISO 8601 format if the site provides it), or "".
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
        if segment in path:
            return category
    return ""
