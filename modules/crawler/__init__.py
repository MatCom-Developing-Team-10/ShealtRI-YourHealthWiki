"""Crawler module for medical data acquisition.

Public API
----------
CrawlConfig   — parameters for a crawl session (delays, limits, output dir)
CrawlResult   — statistics produced at the end of a crawl session
BaseScraper   — ABC that every site-specific scraper must implement

Internal components (not exported here, imported directly when needed):
    modules.crawler.crawler   — GenericCrawler (fetching, queue, robots.txt)
    modules.crawler.registry  — ScraperRegistry (domain → scraper mapping)
    modules.crawler.service   — CrawlerService (single entry point)
    modules.crawler.scrapers  — concrete scraper implementations
"""

from modules.crawler.base import BaseScraper
from modules.crawler.models import CrawlConfig, CrawlResult

__all__ = ["BaseScraper", "CrawlConfig", "CrawlResult"]
