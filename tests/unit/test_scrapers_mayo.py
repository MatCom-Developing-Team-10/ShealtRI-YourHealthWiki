"""Tests for the Mayo Clinic scraper."""

from __future__ import annotations

from modules.crawler.scrapers.mayo_clinic import MayoClinicScraper


class TestCanHandle:
    def test_accepts_diseases_conditions(self):
        s = MayoClinicScraper()
        assert s.can_handle("https://www.mayoclinic.org/diseases-conditions/asthma/symptoms/foo") is True

    def test_accepts_spanish_diseases(self):
        s = MayoClinicScraper()
        assert s.can_handle("https://www.mayoclinic.org/es/diseases-conditions/foo") is True

    def test_rejects_appointment_pages(self):
        s = MayoClinicScraper()
        assert s.can_handle("https://www.mayoclinic.org/appointments/") is False

    def test_rejects_career_pages(self):
        s = MayoClinicScraper()
        assert s.can_handle("https://www.mayoclinic.org/careers") is False


class TestScrape:
    def _html(self, lang="en"):
        return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta name="article:modified_time" content="2025-01-15T10:00:00Z">
</head>
<body>
  <div id="main-content">
    <h1>Hipertensión arterial</h1>
    <p>La hipertensión arterial es una enfermedad cardiovascular crónica que puede causar complicaciones graves si no se trata adecuadamente.</p>
    <p>Los síntomas pueden ser silenciosos durante años.</p>
  </div>
</body>
</html>"""

    def test_extracts_full_document(self):
        s = MayoClinicScraper()
        url = "https://www.mayoclinic.org/diseases-conditions/hipertension"
        doc = s.scrape(url, self._html("es"))
        assert doc is not None
        assert doc.url == url
        assert "hipertensión" in doc.text.lower()
        # Required metadata keys per BaseScraper contract
        assert doc.metadata["title"] == "Hipertensión arterial"
        assert doc.metadata["source"] == "mayo_clinic"
        assert doc.metadata["language"] == "es"
        assert doc.metadata["date"] == "2025-01-15T10:00:00Z"
        assert doc.metadata["category"] == "disease"

    def test_returns_none_when_no_h1(self):
        s = MayoClinicScraper()
        html = "<html lang='en'><body><div id='main-content'><p>No title here</p></div></body></html>"
        assert s.scrape("https://www.mayoclinic.org/diseases-conditions/x", html) is None

    def test_returns_none_when_content_too_short(self):
        s = MayoClinicScraper()
        html = "<html lang='en'><body><div id='main-content'><h1>T</h1><p>tiny</p></div></body></html>"
        assert s.scrape("https://www.mayoclinic.org/diseases-conditions/x", html) is None

    def test_category_inferred_from_url(self):
        s = MayoClinicScraper()
        url = "https://www.mayoclinic.org/symptoms/headache/basics/definition/sym-12345"
        doc = s.scrape(url, self._html("en"))
        assert doc is not None
        assert doc.metadata["category"] == "symptom"

    def test_normalizes_language_with_region_code(self):
        s = MayoClinicScraper()
        # 'es-MX' must be normalized to 'es'
        html = self._html("es-MX")
        doc = s.scrape("https://www.mayoclinic.org/diseases-conditions/x", html)
        assert doc is not None
        assert doc.metadata["language"] == "es"
