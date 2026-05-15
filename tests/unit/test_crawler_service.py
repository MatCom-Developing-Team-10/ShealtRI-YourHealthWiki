"""Tests for CrawlerService — the user-facing facade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modules.crawler.base import BaseScraper
from modules.crawler.models import CrawlConfig, CrawlResult
from modules.crawler.service import CrawlerService


class _DummyScraper(BaseScraper):
    domain = "dummy.test"
    source_name = "dummy"

    def scrape(self, url, html):
        return None


class TestCrawlerService:
    def test_rejects_empty_scraper_list(self, tmp_path):
        with pytest.raises(ValueError, match="(?i)at least one scraper"):
            CrawlerService(scrapers=[], config=CrawlConfig(output_dir=str(tmp_path)))

    def test_uses_default_config_when_none(self, tmp_path, monkeypatch):
        # Avoid filesystem side effects on the default 'data/raw' path
        monkeypatch.chdir(tmp_path)
        service = CrawlerService(scrapers=[_DummyScraper()])
        assert isinstance(service._config, CrawlConfig)

    def test_run_delegates_to_generic_crawler(self, tmp_path):
        config = CrawlConfig(output_dir=str(tmp_path / "raw"))
        service = CrawlerService(scrapers=[_DummyScraper()], config=config)
        with patch(
            "modules.crawler.service.GenericCrawler"
        ) as MockCrawler:
            mock_instance = MagicMock()
            mock_instance.crawl.return_value = CrawlResult(
                documents_saved=0,
                total_visited=0,
                total_successful=0,
                total_failed=0,
            )
            MockCrawler.return_value = mock_instance

            result = service.run()

        MockCrawler.assert_called_once()
        mock_instance.crawl.assert_called_once()
        assert isinstance(result, CrawlResult)
