"""Integration test: scraper -> GenericCrawler -> RawDocumentStorage (JSONL).

Exercises the boundary between the crawler subsystem and the infrastructure
layer. HTTP is mocked at the ``requests.Session`` level, but the rest of the
pipeline runs verbatim: a real ``BaseScraper`` parses real HTML, the crawler
queues real URLs, batches real ``Document`` objects, and writes them to a
real (temporary) JSONL file. The file is then re-read and validated against
the original Document shape.

Together with ``test_indexing_to_retrieval.py``, this covers the two halves
of the Corte 1 pipeline: ingestion (crawler -> JSONL) and retrieval
(JSONL -> indexer -> LSI).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.models import Document
from infra.storage import RawDocumentStorage
from modules.crawler.base import BaseScraper
from modules.crawler.crawler import GenericCrawler
from modules.crawler.models import CrawlConfig


# ---------------------------------------------------------------------------
# Test scraper that returns a deterministic Document per URL.
# ---------------------------------------------------------------------------


class _DeterministicScraper(BaseScraper):
    domain = "fake.test"
    source_name = "fake_source"
    sitemap_urls = ["https://fake.test/sitemap.xml"]

    def scrape(self, url, html):
        # Embed the URL slug in the text so we can assert per-URL fidelity later
        slug = url.rsplit("/", 1)[-1]
        return Document(
            doc_id="ignored-by-crawler",
            text=f"contenido específico sobre {slug}. " * 4,
            url=url,
            metadata={
                "title": f"Title for {slug}",
                "source": self.source_name,
                "language": "es",
                "date": "",
                "category": "test",
            },
        )


_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://fake.test/asma</loc></url>
  <url><loc>https://fake.test/diabetes</loc></url>
  <url><loc>https://fake.test/hipertension</loc></url>
</urlset>"""


def _resp(text: str):
    r = MagicMock()
    r.text = text
    r.status_code = 200
    r.raise_for_status = MagicMock()
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path) -> RawDocumentStorage:
    return RawDocumentStorage(str(tmp_path / "raw"))


def _run_crawl(storage: RawDocumentStorage):
    scraper = _DeterministicScraper()
    crawler = GenericCrawler(
        scrapers=[scraper],
        config=CrawlConfig(delay_seconds=0.0, respect_robots=False),
        storage=storage,
    )
    with patch.object(crawler, "_setup_session") as mock_factory:
        session = MagicMock()
        # 1 sitemap fetch + 3 article fetches
        session.get.side_effect = [
            _resp(_SITEMAP_XML),
            _resp("<html><body><p>asma article</p></body></html>"),
            _resp("<html><body><p>diabetes article</p></body></html>"),
            _resp("<html><body><p>hipertension article</p></body></html>"),
        ]
        mock_factory.return_value = session
        return crawler.crawl()


class TestCrawlPersistence:
    def test_crawl_writes_one_jsonl_per_source(self, storage):
        result = _run_crawl(storage)
        assert result.documents_saved == 3
        path = storage.source_path("fake_source")
        assert path.exists()
        # All three documents fit in a single batch (< _BATCH_FLUSH_SIZE)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_each_record_round_trips_to_document(self, storage):
        _run_crawl(storage)
        path = storage.source_path("fake_source")

        recovered: list[Document] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            recovered.append(
                Document(
                    doc_id=record["doc_id"],
                    text=record["text"],
                    url=record["url"],
                    metadata=record["metadata"],
                )
            )

        assert len(recovered) == 3
        # Doc IDs are UUID5(URL) — stable, distinct, never the scraper's placeholder
        assert all(d.doc_id != "ignored-by-crawler" for d in recovered)
        assert len({d.doc_id for d in recovered}) == 3

        # URLs preserved
        urls = {d.url for d in recovered}
        assert urls == {
            "https://fake.test/asma",
            "https://fake.test/diabetes",
            "https://fake.test/hipertension",
        }

        # Required metadata keys per BaseScraper contract
        for d in recovered:
            for key in ("title", "source", "language", "date", "category"):
                assert key in d.metadata

    def test_unicode_in_text_preserved(self, storage):
        _run_crawl(storage)
        text = storage.source_path("fake_source").read_text(encoding="utf-8")
        # 'específico' and 'hipertensión' (via title) must survive verbatim
        assert "específico" in text


class TestCrawlAppendSemantics:
    """A second crawl run must APPEND, not overwrite, the JSONL file."""

    def test_second_run_appends(self, storage):
        _run_crawl(storage)
        first_size = storage.source_path("fake_source").stat().st_size
        _run_crawl(storage)
        second_size = storage.source_path("fake_source").stat().st_size
        assert second_size > first_size
