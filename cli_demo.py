"""CLI demo — same interface as cli.py, no external dependencies.

    python cli_demo.py
    python cli_demo.py --query "diabetes tipo 2"
    python cli_demo.py --stats

Uses a stub pipeline backed by the 20 synthetic medical documents and a
keyword-overlap scoring function instead of LSI. The displayed output is
identical to what cli.py produces with the real pipeline.

Use this to verify the interface behavior when Docker is not available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Stub heavy deps so core.models / indexer can be imported
# ---------------------------------------------------------------------------
import types
from unittest.mock import MagicMock

def _pkg(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod

_Language = type("Language", (), {})
_spacy_language = _pkg("spacy.language", Language=_Language)
_spacy = _pkg("spacy", language=_spacy_language, Language=_Language)
_spacy.load = MagicMock(return_value=MagicMock(return_value=MagicMock(__iter__=lambda s: iter([]))))
sys.modules.update({
    "spacy": _spacy,
    "spacy.language": _spacy_language,
    "spacy.lang": _pkg("spacy.lang"),
    "spacy.lang.es": _pkg("spacy.lang.es"),
})
_nltk_corpus = _pkg("nltk.corpus")
_nltk_corpus.stopwords = MagicMock()
_nltk_corpus.stopwords.words = MagicMock(return_value=[])
_nltk = _pkg("nltk", corpus=_nltk_corpus)
_nltk.download = MagicMock()
sys.modules.update({"nltk": _nltk, "nltk.corpus": _nltk_corpus})
for _m in [
    "joblib", "chromadb",
    "sklearn", "sklearn.decomposition",
    "sklearn.feature_extraction", "sklearn.feature_extraction.text",
    "langchain", "langchain.schema",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.document_loaders.base",
    "sentence_transformers",
]:
    sys.modules.setdefault(_m, MagicMock())

# ---------------------------------------------------------------------------
# Now safe to import project code
# ---------------------------------------------------------------------------
from core.models import Document, Query, RetrievedDocument
from modules.indexer.service import IndexerService
from tests._synthetic_corpus import RAW_DOCUMENTS

# ---------------------------------------------------------------------------
# Stub TextProcessor and stub pipeline
# ---------------------------------------------------------------------------

class _StubTextProcessor:
    """Tokenise by lowercasing and splitting — no NLP, no spaCy."""

    def process(self, text: str, is_query: bool = False) -> str:
        tokens = []
        for raw in text.lower().split():
            tok = raw.strip(".,;:!?\"'()[]{}")
            if len(tok) >= 3:
                tokens.append(tok)
        return " ".join(tokens)


def _keyword_score(query_tokens: set[str], doc_text: str) -> float:
    """Overlap coefficient between query tokens and document text tokens."""
    if not query_tokens:
        return 0.0
    doc_tokens = set(doc_text.lower().split())
    overlap = len(query_tokens & doc_tokens)
    return round(overlap / len(query_tokens), 3)


class StubPipeline:
    """Same public API as cli.Pipeline — powered by keyword overlap instead of LSI."""

    def __init__(self) -> None:
        self._processor = _StubTextProcessor()
        self._indexer = IndexerService(text_processor=self._processor)
        self._documents: list[Document] = []
        self._corpus = None

    def build(self) -> None:
        self._documents = [
            Document(
                doc_id=d["doc_id"],
                text=d["text"],
                url=d["url"],
                metadata={"title": d["title"]},
            )
            for d in RAW_DOCUMENTS
        ]
        print(f"  source  : synthetic corpus ({len(self._documents)} docs)")
        print("  indexing...", end=" ", flush=True)
        self._corpus = self._indexer.build(self._documents)
        stats = IndexerService.stats(self._corpus)
        print(f"done  [{stats['n_documents']} docs, {stats['n_terms']} terms]")
        print("  fitting stub scorer (keyword overlap)... done")

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievedDocument]:
        processed = self._processor.process(query_text, is_query=True)
        query_tokens = set(processed.split())
        scored = [
            RetrievedDocument(document=doc, score=_keyword_score(query_tokens, doc.text))
            for doc in self._documents
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return [r for r in scored[:top_k] if r.score > 0.0]

    def stats(self) -> dict:
        if self._corpus is None:
            return {}
        return IndexerService.stats(self._corpus)


# ---------------------------------------------------------------------------
# Display helpers — identical to cli.py
# ---------------------------------------------------------------------------

_BANNER = """
╔══════════════════════════════════════════════════╗
║      ShealtRI — Medical Information SRI          ║
║      [DEMO MODE — keyword overlap scorer]        ║
║   Type a query, 'stats', 'help', or 'quit'       ║
╚══════════════════════════════════════════════════╝"""


def _print_results(results: list, query_text: str) -> None:
    if not results:
        print(f"\n  No results found for: '{query_text}'")
        return
    print(f"\n  Results ({len(results)} found):")
    for i, r in enumerate(results, start=1):
        title = r.document.metadata.get("title", r.document.doc_id)
        url = r.document.url or "(no url)"
        snippet = r.document.text[:120].replace("\n", " ") + "..."
        print(f"\n  {i}. [{r.score:.3f}] {title}")
        print(f"       {url}")
        print(f"       {snippet}")


def _print_stats(stats: dict) -> None:
    if not stats:
        print("  Pipeline not loaded yet.")
        return
    print()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:<25}: {value:.2f}")
        else:
            print(f"  {key:<25}: {value}")


def _print_help() -> None:
    print("""
  Commands:
    <query>    Search for medical information
    stats      Show corpus statistics
    help       Show this help message
    quit       Exit the program
    """)


# ---------------------------------------------------------------------------
# Entry points — identical to cli.py
# ---------------------------------------------------------------------------

def run_interactive(pipeline: StubPipeline) -> None:
    print(_BANNER)
    print()
    try:
        while True:
            try:
                raw = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not raw:
                continue
            if raw.lower() in {"quit", "exit", "q"}:
                print("Bye.")
                break
            if raw.lower() == "stats":
                _print_stats(pipeline.stats())
                continue
            if raw.lower() in {"help", "?"}:
                _print_help()
                continue

            results = pipeline.retrieve(raw)
            _print_results(results, raw)
            print()

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


def run_oneshot(pipeline: StubPipeline, query: str) -> None:
    results = pipeline.retrieve(query)
    _print_results(results, query)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShealtRI Demo — Medical IR (no external deps)",
    )
    parser.add_argument("--query", "-q", help="Run a single query and exit")
    parser.add_argument("--stats", action="store_true", help="Print corpus stats and exit")
    parser.add_argument("--top-k", type=int, default=5, metavar="K")
    args = parser.parse_args()

    print("[ShealtRI DEMO] Loading pipeline...")
    pipeline = StubPipeline()
    pipeline.build()
    print()

    if args.stats:
        _print_stats(pipeline.stats())
    elif args.query:
        run_oneshot(pipeline, args.query)
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
