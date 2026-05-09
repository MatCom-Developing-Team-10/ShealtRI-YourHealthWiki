"""Tests for the MedlinePlus scraper."""

from __future__ import annotations

from modules.crawler.scrapers.medlineplus import MedlinePlusScraper


class TestCanHandle:
    def test_accepts_root_health_topic(self):
        s = MedlinePlusScraper()
        assert s.can_handle("https://medlineplus.gov/diabetes.html") is True

    def test_accepts_spanish_health_topic(self):
        s = MedlinePlusScraper()
        assert s.can_handle("https://medlineplus.gov/spanish/diabetes.html") is True

    def test_accepts_encyclopedia_article(self):
        s = MedlinePlusScraper()
        assert s.can_handle("https://medlineplus.gov/ency/article/000305.htm") is True

    def test_rejects_organizations(self):
        s = MedlinePlusScraper()
        assert s.can_handle("https://medlineplus.gov/organizations/") is False

    def test_rejects_druginfo(self):
        s = MedlinePlusScraper()
        assert s.can_handle("https://medlineplus.gov/druginfo/meds/a682530.html") is False


class TestScrape:
    _HEALTH_TOPIC_HTML = """<html lang="en">
<head><meta name="DC.Date.Modified" content="2025-03-21"></head>
<body>
<article>
  <h1 class="with-also">Diabetes</h1>
  <section id="topsum_section">
    <p>Diabetes is a disease in which your blood glucose levels are too high. Glucose comes from the foods you eat.</p>
    <p>The hormone insulin helps glucose get into your cells to give them energy.</p>
  </section>
</article>
</body>
</html>"""

    def test_extracts_health_topic(self):
        s = MedlinePlusScraper()
        doc = s.scrape("https://medlineplus.gov/diabetes.html", self._HEALTH_TOPIC_HTML)
        assert doc is not None
        assert doc.metadata["title"] == "Diabetes"
        assert doc.metadata["source"] == "medlineplus"
        assert doc.metadata["language"] == "en"
        assert doc.metadata["date"] == "2025-03-21"
        assert doc.metadata["category"] == "health-topic"
        assert "glucose" in doc.text

    def test_extracts_encyclopedia_with_fallback(self):
        s = MedlinePlusScraper()
        html = """<html lang="en">
<body><article><h1>Article</h1>
  <p>Encyclopedia entries describe a single condition with detailed clinical content.</p>
  <p>This is the second paragraph of the entry.</p>
</article></body></html>"""
        doc = s.scrape("https://medlineplus.gov/ency/article/000305.htm", html)
        assert doc is not None
        assert doc.metadata["category"] == "reference"

    def test_returns_none_when_no_title(self):
        s = MedlinePlusScraper()
        html = "<html><body><section id='topsum_section'><p>x</p></section></body></html>"
        assert s.scrape("https://medlineplus.gov/x.html", html) is None
