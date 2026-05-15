"""Tests for GenericCrawler (HTTP, queue, robots, persistence).

External I/O is fully mocked: no real HTTP, no real disk writes for HTML.
``RawDocumentStorage`` is wired through pytest's ``tmp_path`` so we exercise
the persistence boundary in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import Document
from infra.storage import RawDocumentStorage
from modules.crawler.base import BaseScraper
from modules.crawler.crawler import GenericCrawler
from modules.crawler.models import CrawlConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeScraper(BaseScraper):
    domain = "fake.test"
    source_name = "fake"
    sitemap_urls = ["https://fake.test/sitemap.xml"]

    def __init__(self, return_doc: bool = True) -> None:
        self.return_doc = return_doc
        self.scrape_calls: list[str] = []

    def scrape(self, url, html):
        self.scrape_calls.append(url)
        if not self.return_doc:
            return None
        return Document(
            doc_id="placeholder",  # crawler will overwrite with UUID5
            text="extracted text " * 5,
            url=url,
            metadata={"title": "T", "language": "en", "category": "x"},
        )


_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://fake.test/article1</loc></url>
  <url><loc>https://fake.test/article2</loc></url>
</urlset>"""


def _mock_response(text: str, status: int = 200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDocIdGeneration:
    def test_same_url_yields_same_id(self):
        a = GenericCrawler._generate_doc_id("https://x.com/page")
        b = GenericCrawler._generate_doc_id("https://x.com/page")
        assert a == b

    def test_different_urls_yield_different_ids(self):
        a = GenericCrawler._generate_doc_id("https://x.com/a")
        b = GenericCrawler._generate_doc_id("https://x.com/b")
        assert a != b


class TestCrawlEndToEnd:
    def test_crawl_persists_documents(self, tmp_path):
        scraper = _FakeScraper()
        storage = RawDocumentStorage(str(tmp_path / "raw"))
        crawler = GenericCrawler(
            scrapers=[scraper],
            config=CrawlConfig(
                delay_seconds=0.0,
                respect_robots=False,
                max_pages=None,
            ),
            storage=storage,
        )

        with patch.object(crawler, "_setup_session") as mock_session_factory:
            session = MagicMock()
            # First call: sitemap fetch; following calls: article HTML
            session.get.side_effect = [
                _mock_response(_SITEMAP_XML),
                _mock_response("<html><body><p>article 1</p></body></html>"),
                _mock_response("<html><body><p>article 2</p></body></html>"),
            ]
            mock_session_factory.return_value = session

            result = crawler.crawl()

        assert result.total_visited == 2
        assert result.total_successful == 2
        assert result.documents_saved == 2
        # Both pages were passed to scraper.scrape
        assert len(scraper.scrape_calls) == 2
        # Storage file exists with the persisted batch
        path = storage.source_path("fake")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_max_pages_limit_respected(self, tmp_path):
        scraper = _FakeScraper()
        storage = RawDocumentStorage(str(tmp_path / "raw"))
        crawler = GenericCrawler(
            scrapers=[scraper],
            config=CrawlConfig(
                delay_seconds=0.0, respect_robots=False, max_pages=1
            ),
            storage=storage,
        )

        with patch.object(crawler, "_setup_session") as mock_session_factory:
            session = MagicMock()
            session.get.side_effect = [
                _mock_response(_SITEMAP_XML),
                _mock_response("<html><body><p>art</p></body></html>"),
            ]
            mock_session_factory.return_value = session

            result = crawler.crawl()

        assert result.total_visited == 1

    def test_scraper_returning_none_counts_as_failure(self, tmp_path):
        scraper = _FakeScraper(return_doc=False)
        storage = RawDocumentStorage(str(tmp_path / "raw"))
        crawler = GenericCrawler(
            scrapers=[scraper],
            config=CrawlConfig(delay_seconds=0.0, respect_robots=False),
            storage=storage,
        )

        with patch.object(crawler, "_setup_session") as mock_session_factory:
            session = MagicMock()
            session.get.side_effect = [
                _mock_response(_SITEMAP_XML),
                _mock_response("<html></html>"),
                _mock_response("<html></html>"),
            ]
            mock_session_factory.return_value = session

            result = crawler.crawl()

        assert result.total_successful == 0
        assert result.total_failed == 2
        assert result.documents_saved == 0

    def test_scraper_exception_does_not_abort(self, tmp_path):
        class _CrashingScraper(_FakeScraper):
            def scrape(self, url, html):
                if "1" in url:
                    raise RuntimeError("boom")
                return super().scrape(url, html)

        scraper = _CrashingScraper()
        storage = RawDocumentStorage(str(tmp_path / "raw"))
        crawler = GenericCrawler(
            scrapers=[scraper],
            config=CrawlConfig(delay_seconds=0.0, respect_robots=False),
            storage=storage,
        )

        with patch.object(crawler, "_setup_session") as mock_session_factory:
            session = MagicMock()
            session.get.side_effect = [
                _mock_response(_SITEMAP_XML),
                _mock_response("<html></html>"),
                _mock_response("<html></html>"),
            ]
            mock_session_factory.return_value = session

            result = crawler.crawl()

        # 1 succeeded (article2), 1 failed (article1 crashed)
        assert result.total_successful == 1
        assert result.total_failed == 1
        assert any("boom" in err for _, err in result.errors)


class TestSitemapIndexRecursion:
    def test_sitemapindex_expands_to_child_sitemaps(self, tmp_path):
        scraper = _FakeScraper()
        storage = RawDocumentStorage(str(tmp_path / "raw"))
        crawler = GenericCrawler(
            scrapers=[scraper],
            config=CrawlConfig(delay_seconds=0.0, respect_robots=False),
            storage=storage,
        )

        sitemap_index = """<?xml version="1.0"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap><loc>https://fake.test/child.xml</loc></sitemap>
        </sitemapindex>"""

        with patch.object(crawler, "_setup_session") as mock_session_factory:
            session = MagicMock()
            session.get.side_effect = [
                _mock_response(sitemap_index),
                _mock_response(_SITEMAP_XML),
                _mock_response("<html></html>"),
                _mock_response("<html></html>"),
            ]
            mock_session_factory.return_value = session
            result = crawler.crawl()

        assert result.total_visited == 2
