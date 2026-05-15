"""Tests for the BaseScraper contract."""

from __future__ import annotations

import pytest

from core.models import Document
from modules.crawler.base import BaseScraper


class _ValidScraper(BaseScraper):
    domain = "example.org"
    source_name = "example"
    sitemap_urls = ["https://example.org/sitemap.xml"]

    def scrape(self, url: str, html: str) -> Document | None:
        return Document(doc_id="x", text="content", url=url)


class TestSubclassEnforcement:
    def test_missing_domain_raises(self):
        with pytest.raises(TypeError, match="domain"):

            class BadScraper(BaseScraper):  # noqa: F841
                source_name = "x"

                def scrape(self, url, html):
                    return None

    def test_missing_source_name_raises(self):
        with pytest.raises(TypeError, match="source_name"):

            class BadScraper2(BaseScraper):  # noqa: F841
                domain = "x.com"

                def scrape(self, url, html):
                    return None


class TestDefaultBehaviors:
    def test_can_handle_default_uses_substring_match(self):
        scraper = _ValidScraper()
        assert scraper.can_handle("https://example.org/foo") is True
        assert scraper.can_handle("https://other.org/example.org/foo") is True
        assert scraper.can_handle("https://other.org/foo") is False

    def test_get_sitemap_urls_returns_copy(self):
        scraper = _ValidScraper()
        urls = scraper.get_sitemap_urls()
        urls.append("https://malicious")
        # Mutating the returned list must not affect the class attribute
        assert "https://malicious" not in scraper.sitemap_urls
