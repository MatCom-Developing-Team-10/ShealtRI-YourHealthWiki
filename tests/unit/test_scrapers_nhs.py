"""Tests for the NHS scraper."""

from __future__ import annotations

from modules.crawler.scrapers.nhs import NHSScraper


class TestCanHandle:
    def test_accepts_conditions(self):
        s = NHSScraper()
        assert s.can_handle("https://www.nhs.uk/conditions/diabetes/") is True

    def test_accepts_medicines(self):
        s = NHSScraper()
        assert s.can_handle("https://www.nhs.uk/medicines/metformin/") is True

    def test_accepts_live_well(self):
        s = NHSScraper()
        assert s.can_handle("https://www.nhs.uk/live-well/eat-well/") is True

    def test_rejects_service_search(self):
        s = NHSScraper()
        assert s.can_handle("https://www.nhs.uk/service-search/") is False


class TestScrape:
    _HTML = """<html lang="en">
<head>
  <meta name="article:modified_time" content="3 Jul 2025, 11:31 a.m.">
</head>
<body>
  <main id="maincontent">
    <h1>Asthma</h1>
    <p>Asthma is a common lung condition that causes occasional breathing difficulties.</p>
    <p>It affects people of all ages and often starts in childhood.</p>
  </main>
</body>
</html>"""

    def test_extracts_condition_page(self):
        s = NHSScraper()
        doc = s.scrape("https://www.nhs.uk/conditions/asthma/", self._HTML)
        assert doc is not None
        assert doc.metadata["title"] == "Asthma"
        assert doc.metadata["source"] == "nhs"
        assert doc.metadata["language"] == "en"
        assert doc.metadata["category"] == "disease"
        assert doc.metadata["date"].startswith("3 Jul 2025")
        assert "Asthma" in doc.text

    def test_category_mapping(self):
        s = NHSScraper()
        for url, expected in [
            ("https://www.nhs.uk/conditions/x/", "disease"),
            ("https://www.nhs.uk/medicines/y/", "drug"),
            ("https://www.nhs.uk/live-well/z/", "wellness"),
            ("https://www.nhs.uk/mental-health/a/", "mental-health"),
        ]:
            doc = s.scrape(url, self._HTML)
            assert doc is not None
            assert doc.metadata["category"] == expected, url

    def test_returns_none_when_content_too_short(self):
        s = NHSScraper()
        html = "<html><body><main><h1>Title</h1><p>tiny</p></main></body></html>"
        assert s.scrape("https://www.nhs.uk/conditions/x/", html) is None
