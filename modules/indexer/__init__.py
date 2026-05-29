"""Indexer module for document storage, corpus building, and persistence.

Lazy-loading note
-----------------
``FileSystemDocumentStore`` and its error classes are eagerly available
because they have no heavy dependencies. ``IndexerService`` /
``IndexStore`` (and their config / error siblings) are exposed via PEP-562
``__getattr__`` so that the line::

    import modules.indexer

does **not** transitively import spaCy or NLTK via the text-processor
chain. Code that needs the heavy classes still imports them normally::

    from modules.indexer import IndexerService   # triggers the lazy load

The lazy plumbing is invisible to callers — ``dir(modules.indexer)`` and
``hasattr`` work as expected.
"""

from .document_store import (
    DocumentReadError,
    DocumentStoreError,
    DocumentWriteError,
    FileSystemDocumentStore,
)

__all__ = [
    # Document storage (eager — no heavy deps)
    "FileSystemDocumentStore",
    "DocumentStoreError",
    "DocumentWriteError",
    "DocumentReadError",
    # Indexer service (lazy — pulls TextProcessor → spaCy)
    "IndexerService",
    "IndexerConfig",
    # Indexer persistence / management (lazy)
    "IndexStore",
    "IndexStoreError",
]


# Mapping of public name → (submodule, attribute).  Resolved on first access
# and cached in ``globals()`` so subsequent reads are a plain dict lookup.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "IndexerService": ("modules.indexer.service", "IndexerService"),
    "IndexerConfig": ("modules.indexer.service", "IndexerConfig"),
    "IndexStore": ("modules.indexer.index_store", "IndexStore"),
    "IndexStoreError": ("modules.indexer.index_store", "IndexStoreError"),
}


def __getattr__(name: str):
    """Resolve and cache lazy attributes (PEP 562)."""
    if name in _LAZY_ATTRS:
        from importlib import import_module

        mod_name, attr = _LAZY_ATTRS[name]
        resolved = getattr(import_module(mod_name), attr)
        globals()[name] = resolved
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
