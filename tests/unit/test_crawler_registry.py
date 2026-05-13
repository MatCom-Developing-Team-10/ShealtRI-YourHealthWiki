"""Tests for ScraperRegistry."""

from __future__ import annotations

import pytest

from core.models import Document
from modules.crawler.base import BaseScraper
from modules.crawler.registry import ScraperRegistry


class _ScraperA(BaseScraper):
    domain = "a.org"
    source_name = "a"

    def scrape(self, url, html):
        return None


class _ScraperB(BaseScraper):
    domain = "b.org"
    source_name = "b"

    def scrape(self, url, html):
        return None


class TestRegister:
    def test_register_appends_in_order(self):
        reg = ScraperRegistry()
        reg.register(_ScraperA())
        reg.register(_ScraperB())
        assert len(reg) == 2
        assert [type(s).__name__ for s in reg.all_scrapers()] == [
            "_ScraperA",
            "_ScraperB",
        ]

    def test_register_rejects_non_scraper(self):
        reg = ScraperRegistry()
        with pytest.raises(TypeError):
            reg.register(object())  # type: ignore[arg-type]


class TestGet:
    def test_returns_first_matching_scraper(self):
        reg = ScraperRegistry()
        reg.register(_ScraperA())
        reg.register(_ScraperB())
        match = reg.get("https://a.org/page")
        assert isinstance(match, _ScraperA)

    def test_returns_none_when_no_match(self):
        reg = ScraperRegistry()
        reg.register(_ScraperA())
        assert reg.get("https://other.org/page") is None

    def test_registration_order_matters_for_overlap(self):
        # Define two scrapers that both could match a URL.
        class _ScraperOverlapA(BaseScraper):
            domain = "shared"
            source_name = "ovA"

            def scrape(self, url, html):
                return None

        class _ScraperOverlapB(BaseScraper):
            domain = "shared"
            source_name = "ovB"

            def scrape(self, url, html):
                return None

        reg = ScraperRegistry()
        reg.register(_ScraperOverlapB())
        reg.register(_ScraperOverlapA())
        # The first registered wins
        assert isinstance(reg.get("https://shared/page"), _ScraperOverlapB)
