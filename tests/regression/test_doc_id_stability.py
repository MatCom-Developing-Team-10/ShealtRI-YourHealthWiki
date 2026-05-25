"""Regression test for doc_id generation.

The crawler assigns ``doc_id = UUID5(NAMESPACE_URL, url)``. Re-crawling the
same URL must produce the same ID — this is what makes the indexer's
deduplication work and what lets users diff snapshots of the index.

If somebody changes the hashing strategy, every existing index becomes
invalid (no document is "the same document" anymore). This test pins the
current UUID5 strategy with a single golden value.
"""

from __future__ import annotations

from modules.crawler.crawler import GenericCrawler


GOLDEN_DOC_IDS = {
    # The deterministic UUID5(NAMESPACE_URL, url) — DO NOT regenerate unless
    # you are intentionally invalidating every existing index.
    "https://www.mayoclinic.org/diseases-conditions/diabetes": (
        "700af91f-1b83-57f6-8e34-9ee6f53e1efb"
    ),
    "https://medlineplus.gov/diabetes.html": (
        "f39a5964-6442-566f-86bc-0e4a90bc8cab"
    ),
    "https://www.nhs.uk/conditions/asthma/": (
        "0a2f8034-3242-517a-a432-888c95604bbe"
    ),
}


def test_doc_id_is_uuid5_of_url():
    for url, expected_id in GOLDEN_DOC_IDS.items():
        got = GenericCrawler._generate_doc_id(url)
        assert got == expected_id, (
            f"doc_id strategy changed for {url}\n"
            f"  expected: {expected_id}\n"
            f"  got:      {got}\n"
            f"WARNING: changing doc_id generation INVALIDATES every existing index. "
            f"Coordinate with the team before updating this golden table."
        )


def test_doc_id_is_stable_across_invocations():
    url = "https://example.com/some/article"
    a = GenericCrawler._generate_doc_id(url)
    b = GenericCrawler._generate_doc_id(url)
    assert a == b


def test_different_urls_yield_different_ids():
    a = GenericCrawler._generate_doc_id("https://example.com/x")
    b = GenericCrawler._generate_doc_id("https://example.com/y")
    assert a != b
