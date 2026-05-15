"""Tests for crawler models (CrawlConfig, CrawlResult)."""

from __future__ import annotations

import pytest

from modules.crawler.models import CrawlConfig, CrawlResult


class TestCrawlConfig:
    def test_defaults(self):
        cfg = CrawlConfig()
        assert cfg.delay_seconds > 0
        assert cfg.respect_robots is True
        assert cfg.user_agent  # non-empty
        assert cfg.max_pages is None
        assert cfg.output_dir == "data/raw"
        assert cfg.request_timeout == 15

    def test_overrides(self):
        cfg = CrawlConfig(delay_seconds=0.1, max_pages=10, respect_robots=False)
        assert cfg.delay_seconds == 0.1
        assert cfg.max_pages == 10
        assert cfg.respect_robots is False


class TestCrawlResult:
    def test_success_rate_basic(self):
        r = CrawlResult(
            documents_saved=8,
            total_visited=10,
            total_successful=8,
            total_failed=2,
        )
        assert r.success_rate == pytest.approx(0.8)

    def test_success_rate_zero_visited(self):
        r = CrawlResult(
            documents_saved=0,
            total_visited=0,
            total_successful=0,
            total_failed=0,
        )
        assert r.success_rate == 0.0

    def test_str_includes_counters(self):
        r = CrawlResult(
            documents_saved=3,
            total_visited=4,
            total_successful=3,
            total_failed=1,
            duration_seconds=12.34,
        )
        s = str(r)
        assert "saved=3" in s
        assert "visited=4" in s
        assert "ok=3" in s
        assert "failed=1" in s
        assert "12.3" in s

    def test_default_errors_is_empty(self):
        r = CrawlResult(
            documents_saved=0,
            total_visited=0,
            total_successful=0,
            total_failed=0,
        )
        assert r.errors == []
