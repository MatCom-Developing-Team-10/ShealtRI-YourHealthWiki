"""ShealtRI console interface for manual pipeline testing.

Two modes:
    Interactive REPL:
        python cli.py

    One-shot query:
        python cli.py --query "síntomas de hipertensión"

    With user profile:
        python cli.py --query "diabetes tipo 2" --profile estudiante

    Show corpus statistics only:
        python cli.py --stats

The pipeline is loaded once on startup. If JSONL files exist in data/raw/,
they are used as the corpus. Otherwise, 20 built-in synthetic medical
documents are loaded automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap path so the project root is on sys.path when running as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.models import Document, Query, UserProfile, UserProfileType
from core.pipeline import RetrievalContext
from infra.chroma_repository import ChromaRepository
from modules.document_loader.service import DocumentLoader, DocumentLoaderError
from modules.indexer.document_store import FileSystemDocumentStore
from modules.indexer.service import IndexerService
from modules.retriever.service import LSIRetriever
from modules.text_processor.service import TextProcessor
from modules.rag.service import RAGService
from tests._synthetic_corpus import RAW_DOCUMENTS

_CHROMA_DIR = "data/chroma"
_STORE_DIR = "data/documents"
_MODELS_DIR = "models/lsi"
_RAW_DIR = Path("data/raw")

_PROFILE_MAP: dict[str, UserProfileType] = {
    "paciente": UserProfileType.PATIENT,
    "estudiante": UserProfileType.MEDICAL_STUDENT,
    "medico": UserProfileType.MEDICAL_PROFESSIONAL,
    "diagnostico": UserProfileType.DIAGNOSTIC_ASSISTANT,
    "natural": UserProfileType.NATURAL_MEDICINE,
    "cuidador": UserProfileType.CAREGIVER,
}

_BANNER = """
╔══════════════════════════════════════════════════╗
║         ShealtRI — Medical Information SRI       ║
║   Type a query, 'stats', 'help', or 'quit'       ║
╚══════════════════════════════════════════════════╝"""


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _load_from_raw_dir() -> list[Document] | None:
    """Load documents from data/raw/ — JSONL (crawler output) and PDF/TXT/etc."""
    if not _RAW_DIR.exists():
        return None

    documents: list[Document] = []

    # JSONL files (crawler output format)
    for jsonl_file in sorted(_RAW_DIR.glob("*.jsonl")):
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    documents.append(Document(
                        doc_id=str(data["doc_id"]),
                        text=str(data["text"]),
                        url=str(data.get("url", "")),
                        metadata=data.get("metadata", {}),
                    ))
        except (json.JSONDecodeError, KeyError):
            continue

    # PDF, TXT, JSON, CSV, Markdown — via DocumentLoader
    supported_extensions = {".pdf", ".txt", ".json", ".csv", ".md"}
    non_jsonl_files = [
        f for f in _RAW_DIR.rglob("*")
        if f.is_file() and f.suffix in supported_extensions
    ]
    if non_jsonl_files:
        loader = DocumentLoader()
        try:
            loaded = loader.load_from_directory(_RAW_DIR)
            documents.extend(loaded)
        except DocumentLoaderError as e:
            print(f"  [warn] DocumentLoader: {e}", file=sys.stderr)

    return documents if documents else None


def _load_synthetic_documents() -> list[Document]:
    """Return the 20 built-in synthetic medical documents."""
    return [
        Document(
            doc_id=d["doc_id"],
            text=d["text"],
            url=d["url"],
            metadata={"title": d["title"]},
        )
        for d in RAW_DOCUMENTS
    ]


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

class Pipeline:
    """Wires and holds the full retrieval pipeline."""

    def __init__(self) -> None:
        self.text_processor = None
        self.indexer = None
        self.repository = ChromaRepository(
            persist_directory=_CHROMA_DIR,
            collection_name="medical_documents",
        )
        self.document_store = FileSystemDocumentStore(storage_dir=_STORE_DIR)
        self.retriever = LSIRetriever(
            repository=self.repository,
            document_store=self.document_store,
            model_dir=_MODELS_DIR,
        )
        self.context = RetrievalContext(strategy=self.retriever)
        self.rag_service = RAGService()
        self.corpus = None
        self._source_label = ""

    def build(self) -> None:
        """Load documents, build index, and fit the LSI model."""
        print("  loading NLP model (spaCy)...", end=" ", flush=True)
        self.text_processor = TextProcessor()
        self.indexer = IndexerService(text_processor=self.text_processor)
        print("done")

        print("  reading documents from data/raw/...", end=" ", flush=True)
        real_docs = _load_from_raw_dir()
        if real_docs:
            documents = real_docs
            self._source_label = f"data/raw/ ({len(documents)} docs)"
        else:
            documents = _load_synthetic_documents()
            self._source_label = f"synthetic corpus ({len(documents)} docs)"
        print("done")

        print(f"  source  : {self._source_label}")
        print("  indexing...", end=" ", flush=True)
        self.corpus = self.indexer.build(documents)
        print("done")

        stats = IndexerService.stats(self.corpus)
        n_docs = stats["n_documents"]
        n_terms = stats["n_terms"]

        print(f"  fitting LSI (n_components=100)...", end=" ", flush=True)
        self.retriever.fit(self.corpus)
        print(f"done  [{n_docs} docs, {n_terms} terms]")

    def retrieve(
        self, query_text: str, top_k: int = 5, user_profile: UserProfile | None = None
    ) -> tuple[list, object | None]:
        """Run the full query pipeline and return retrieved documents and RAG response.

        Returns:
            Tuple of (retrieved_documents, rag_response).
            rag_response is None if the RAG service fails.
        """
        query_corpus = self.indexer.build_query(query_text)
        query = Query(text=query_text, indexed_corpus=query_corpus, user_profile=user_profile)
        results = self.context.execute_search(query, top_k=top_k)

        rag_response = None
        if results:
            try:
                rag_response = self.rag_service.generate(query, results)
            except Exception as exc:
                print(f"  [warn] RAG generation failed: {exc}", file=sys.stderr)

        return results, rag_response

    def stats(self) -> dict:
        """Return corpus statistics."""
        if self.corpus is None:
            return {}
        return IndexerService.stats(self.corpus)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

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


def _print_rag_response(response: object) -> None:
    """Display a RAG-generated response with profile metadata."""
    if not response or not hasattr(response, 'answer'):
        return

    profile_label = response.profile_type.value.replace("_", " ").title()
    backend = response.model_name if response.used_llm else "plantilla (sin LLM)"
    print(f"\n  ──── Respuesta generada [{profile_label} | {backend}] ────")
    print()

    for line in response.answer.splitlines():
        if line.strip():
            wrapped = textwrap.fill(
                line, width=78, initial_indent="  ", subsequent_indent="  "
            )
            print(wrapped)
        else:
            print()


def _print_help() -> None:
    print("""
  Commands:
    <query>    Search for medical information
    stats      Show corpus statistics
    help       Show this help message
    quit       Exit the program
    """)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_interactive(pipeline: Pipeline, user_profile: UserProfile | None = None) -> None:
    print(_BANNER)
    if user_profile:
        print(f"  Perfil: {user_profile.name}")
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

            results, rag_response = pipeline.retrieve(raw, user_profile=user_profile)
            _print_results(results, raw)
            if rag_response:
                _print_rag_response(rag_response)
            print()

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


def run_oneshot(pipeline: Pipeline, query: str, user_profile: UserProfile | None = None) -> None:
    results, rag_response = pipeline.retrieve(query, user_profile=user_profile)
    _print_results(results, query)
    if rag_response:
        _print_rag_response(rag_response)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShealtRI — Medical Information Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", help="Run a single query and exit")
    parser.add_argument("--stats", action="store_true", help="Print corpus stats and exit")
    parser.add_argument(
        "--top-k", type=int, default=5, metavar="K",
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--profile", "-p",
        choices=list(_PROFILE_MAP.keys()),
        default="paciente",
        metavar="PERFIL",
        help=(
            "User profile for RAG-generated responses. "
            "Options: paciente, estudiante, medico, diagnostico, natural, cuidador. "
            "(default: paciente)"
        ),
    )
    args = parser.parse_args()

    print("[ShealtRI] Loading pipeline...")
    pipeline = Pipeline()
    pipeline.build()
    print()

    user_profile = UserProfile(
        profile_type=_PROFILE_MAP[args.profile],
        name=args.profile.capitalize(),
    ) if args.profile != "paciente" else UserProfile(
        profile_type=UserProfileType.PATIENT,
        name="Paciente",
    )

    if args.stats:
        _print_stats(pipeline.stats())
    elif args.query:
        run_oneshot(pipeline, args.query, user_profile=user_profile)
    else:
        run_interactive(pipeline, user_profile=user_profile)


if __name__ == "__main__":
    main()
