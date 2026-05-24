"""Web search module for the SRI pipeline.

Provides two retrieval strategies:

    WebSearchRetriever    — keyword-based fallback over the *local* corpus.
                            Fast, no network required.

    InternetSearchRetriever — searches DuckDuckGo and downloads real pages.
                              Used when local corpus lacks enough information.

    WebContentFetcher     — low-level component used by InternetSearchRetriever.
                            Can be used standalone for web content acquisition.

Typical wiring (see FallbackRetriever in modules.retriever):
    fetcher = WebContentFetcher()
    internet = InternetSearchRetriever(fetcher, document_store=cache)
    retriever = FallbackRetriever(primary=lsi_retriever, fallback=internet, min_results=3)
"""

from .fetcher import WebContentFetcher
from .internet_retriever import InternetSearchRetriever
from .service import WebSearchRetriever

__all__ = [
    "WebSearchRetriever",
    "WebContentFetcher",
    "InternetSearchRetriever",
]
