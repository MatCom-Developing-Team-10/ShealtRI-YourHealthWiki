"""Tests for ``modules.web_search.fetcher.WebContentFetcher``.

The fetcher does HTTP I/O and uses the ``ddgs`` library for DuckDuckGo
search. Both are mocked end-to-end so the tests are deterministic and run
offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from modules.web_search.fetcher import WebContentFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(status: int = 200, text: str = "", content_type: str = "text/html"):
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.headers = {"Content-Type": content_type}
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    return r


_PAGE_HTML = """
<html><body>
  <header>nav noise</header>
  <article>
    <p>La hipertensión arterial es una enfermedad cardiovascular crónica.</p>
    <p>El tratamiento incluye cambios de estilo de vida y medicamentos.</p>
    <script>tracker();</script>
  </article>
  <footer>not content</footer>
</body></html>
""".strip() + " " + ("texto adicional " * 20)


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_query_returns_empty(self):
        fetcher = WebContentFetcher()
        assert fetcher.fetch("") == []

    def test_whitespace_query_returns_empty(self):
        fetcher = WebContentFetcher()
        assert fetcher.fetch("    ") == []


# ---------------------------------------------------------------------------
# Doc ID generation
# ---------------------------------------------------------------------------


class TestDocIdGeneration:
    def test_doc_id_starts_with_web_prefix(self):
        out = WebContentFetcher._make_doc_id("https://example.org/x")
        assert out.startswith("web_")

    def test_same_url_gives_same_id(self):
        url = "https://example.org/a"
        assert WebContentFetcher._make_doc_id(url) == WebContentFetcher._make_doc_id(url)

    def test_different_urls_give_different_ids(self):
        a = WebContentFetcher._make_doc_id("https://example.org/a")
        b = WebContentFetcher._make_doc_id("https://example.org/b")
        assert a != b


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_strips_noise_tags(self):
        fetcher = WebContentFetcher()
        text = fetcher._extract_text(
            "<html><body><script>x</script><p>keep me</p></body></html>"
        )
        assert "keep me" in text
        assert "x" not in text  # script content gone

    def test_prefers_article_over_body(self):
        fetcher = WebContentFetcher()
        html = "<html><body><article>article content</article><p>filler</p></body></html>"
        text = fetcher._extract_text(html)
        assert "article content" in text
        assert "filler" not in text

    def test_collapses_whitespace(self):
        fetcher = WebContentFetcher()
        text = fetcher._extract_text("<html><body><p>a\n\n\n    b</p></body></html>")
        assert text == "a b"

    def test_no_body_returns_empty_string(self):
        fetcher = WebContentFetcher()
        # No <body>, no <article>, no <main>
        assert fetcher._extract_text("<html><head><title>t</title></head></html>") == ""


# ---------------------------------------------------------------------------
# Download path with mocked HTTP
# ---------------------------------------------------------------------------


class TestDownloadText:
    def test_non_html_content_type_skipped(self):
        fetcher = WebContentFetcher()
        fetcher._session = MagicMock()
        fetcher._session.get.return_value = _resp(
            content_type="application/pdf", text="ignored"
        )
        assert fetcher._download_text("https://x") is None

    def test_timeout_returns_none(self):
        fetcher = WebContentFetcher()
        fetcher._session = MagicMock()
        fetcher._session.get.side_effect = requests.exceptions.Timeout()
        assert fetcher._download_text("https://x") is None

    def test_request_exception_returns_none(self):
        fetcher = WebContentFetcher()
        fetcher._session = MagicMock()
        fetcher._session.get.side_effect = requests.exceptions.ConnectionError("offline")
        assert fetcher._download_text("https://x") is None

    def test_http_error_returns_none(self):
        fetcher = WebContentFetcher()
        fetcher._session = MagicMock()
        fetcher._session.get.return_value = _resp(status=503, text="server down")
        assert fetcher._download_text("https://x") is None

    def test_happy_path_returns_extracted_text(self):
        fetcher = WebContentFetcher()
        fetcher._session = MagicMock()
        fetcher._session.get.return_value = _resp(status=200, text=_PAGE_HTML)
        text = fetcher._download_text("https://x")
        assert text is not None
        assert "hipertensión" in text


# ---------------------------------------------------------------------------
# Single-page assembly
# ---------------------------------------------------------------------------


class TestFetchPage:
    def test_too_short_text_returns_none(self):
        fetcher = WebContentFetcher(min_text_length=200)
        with patch.object(fetcher, "_download_text", return_value="short"):
            assert fetcher._fetch_page("https://x", "T", "snippet too", 0) is None

    def test_snippet_used_as_fallback_when_download_fails(self):
        fetcher = WebContentFetcher(min_text_length=10)
        long_snippet = "snippet content " * 20
        with patch.object(fetcher, "_download_text", return_value=None):
            out = fetcher._fetch_page("https://x", "Title", long_snippet, 0)
        assert out is not None
        assert "snippet content" in out.text

    def test_short_snippet_with_failed_download_returns_none(self):
        fetcher = WebContentFetcher(min_text_length=200)
        with patch.object(fetcher, "_download_text", return_value=None):
            assert fetcher._fetch_page("https://x", "T", "tiny", 0) is None

    def test_long_text_is_truncated_at_word_boundary(self):
        fetcher = WebContentFetcher(min_text_length=10, max_text_length=50)
        long_text = "palabra " * 30  # ~240 chars
        with patch.object(fetcher, "_download_text", return_value=long_text):
            doc = fetcher._fetch_page("https://x", "T", "snippet", 0)
        assert doc is not None
        assert len(doc.text) <= 50

    def test_returned_document_carries_required_metadata(self):
        fetcher = WebContentFetcher(min_text_length=10)
        with patch.object(fetcher, "_download_text", return_value="adequate text " * 20):
            doc = fetcher._fetch_page("https://x.com/y", "Title", "snip", 3)
        assert doc is not None
        assert doc.url == "https://x.com/y"
        assert doc.metadata["title"] == "Title"
        assert doc.metadata["source"] == "web_search"
        assert doc.metadata["rank"] == 3


# ---------------------------------------------------------------------------
# End-to-end fetch() with mocked DDG + downloads
# ---------------------------------------------------------------------------


class TestFetchEndToEnd:
    def test_no_search_results_returns_empty(self):
        fetcher = WebContentFetcher()
        with patch.object(fetcher, "_search_duckduckgo", return_value=[]):
            assert fetcher.fetch("q", max_results=3) == []

    def test_dispatches_to_fetch_page_per_result(self):
        fetcher = WebContentFetcher(min_text_length=10)
        results = [
            ("https://a", "Title A", "snip-a " * 10),
            ("https://b", "Title B", "snip-b " * 10),
        ]
        with patch.object(fetcher, "_search_duckduckgo", return_value=results):
            with patch.object(fetcher, "_download_text", return_value="page text " * 30):
                docs = fetcher.fetch("q", max_results=2)
        assert len(docs) == 2
        urls = {d.url for d in docs}
        assert urls == {"https://a", "https://b"}

    def test_pages_that_fail_are_filtered_out(self):
        fetcher = WebContentFetcher(min_text_length=10)
        results = [
            ("https://a", "A", "snippet a " * 10),
            ("https://b", "B", "tiny"),
        ]
        with patch.object(fetcher, "_search_duckduckgo", return_value=results):
            # 'a' downloads ok, 'b' fails and has too-short snippet
            with patch.object(
                fetcher, "_download_text",
                side_effect=["page text " * 30, None],
            ):
                docs = fetcher.fetch("q", max_results=2)
        urls = {d.url for d in docs}
        assert urls == {"https://a"}


# ---------------------------------------------------------------------------
# DuckDuckGo search wrapper
# ---------------------------------------------------------------------------


class TestSearchDuckDuckGo:
    def test_returns_empty_on_ddgs_missing(self):
        fetcher = WebContentFetcher()
        # Force the inner ``from ddgs import DDGS`` to fail
        with patch.dict("sys.modules", {"ddgs": None}):
            assert fetcher._search_duckduckgo("q", 5) == []

    def test_returns_empty_on_search_exception(self):
        fetcher = WebContentFetcher()
        fake_ddgs_module = MagicMock()
        fake_ddgs_module.DDGS.side_effect = RuntimeError("rate limit")
        with patch.dict("sys.modules", {"ddgs": fake_ddgs_module}):
            assert fetcher._search_duckduckgo("q", 5) == []
