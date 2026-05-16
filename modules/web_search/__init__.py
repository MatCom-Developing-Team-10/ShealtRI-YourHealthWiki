"""Web search fallback module for the SRI pipeline.

This module provides document retrieval when the LSI retriever does not return
sufficient results. It performs keyword-based search across the document corpus
to find relevant medical documents.
"""

from .service import WebSearchRetriever

__all__ = ["WebSearchRetriever"]
