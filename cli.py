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
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load .env file for GEMINI_API_KEY and other config

# ---------------------------------------------------------------------------
# Bootstrap path so the project root is on sys.path when running as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.models import Document, PipelineContext, Query, UserProfile, UserProfileType
from core.pipeline import RetrievalContext
from core.plugin_pipeline import PluginPipeline
from infra.chroma_repository import ChromaRepository
from modules.document_loader.service import DocumentLoader, DocumentLoaderError
from modules.indexer.document_store import FileSystemDocumentStore
from modules.indexer.service import IndexerService
from modules.indexer.chunker import TextChunker
from modules.ranker.service import HybridRanker
from modules.retriever.fallback_retriever import FallbackRetriever
from modules.retriever.service import LSIRetriever
from modules.web_search import InternetSearchRetriever, WebContentFetcher
from modules.text_processor.service import TextProcessor
from modules.rag.evaluator import RAGEvaluator
from modules.rag.service import RAGService
from modules.recommender import ContentBasedRecommender
from plugins.expansion.service import QueryExpansionPlugin
from plugins.feedback import JSONLFeedbackStore, RelevanceFeedbackService
_CHROMA_DIR = "data/chroma"
_STORE_DIR = "data/documents"
_MODELS_DIR = "models/lsi"
_RAW_DIR = Path("data/raw")
_FEEDBACK_PATH = "data/feedback.jsonl"


@dataclass(slots=True)
class PipelineResult:
    """Aggregated output of a single retrieval cycle.

    Bundles the retrieved documents, RAG response, and the side-channel
    information (query expansion, RAG quality scores) that the UI needs
    to surface without re-running the pipeline.
    """

    results: list
    rag_response: object | None
    expansion_terms: list[str] = field(default_factory=list)
    rag_quality: dict[str, float] | None = None


def extract_snippet(text: str, query: str, length: int = 200) -> str:
    """Return a query-aware excerpt from text."""
    clean = text.replace("\n", " ")
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms or len(clean) <= length:
        snippet = clean[:length]
        return snippet + "…" if len(clean) > length else snippet

    lower = clean.lower()
    best_pos, best_score = 0, -1
    for i in range(0, len(clean) - length + 1, 30):
        score = sum(lower[i : i + length].count(t) for t in terms)
        if score > best_score:
            best_score, best_pos = score, i

    start = best_pos
    end = min(start + length, len(clean))
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(clean) else ""
    return prefix + clean[start:end].strip() + suffix

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

    # JSON, TXT, CSV, Markdown — via DocumentLoader, loaded file by file.
    # PDFs are intentionally excluded: they are pre-extracted to JSON by
    # scripts/ingest_pdfs.py, so the app loads the cheap JSON and never
    # re-parses a PDF here (even if a .pdf is left in data/raw/).
    supported_extensions = {".txt", ".json", ".csv", ".md"}
    other_files = [
        f for f in _RAW_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    if other_files:
        loader = DocumentLoader()
        for file_path in other_files:
            try:
                documents.extend(loader.load_from_file(file_path))
            except DocumentLoaderError as e:
                print(f"  [warn] DocumentLoader ({file_path.name}): {e}", file=sys.stderr)

    return documents if documents else None


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
        # Rocchio relevance feedback: persistent JSONL store + service. The
        # retriever consults the service on every query and re-weights the
        # latent vector when judgments exist for the same query text.
        self.feedback_service = RelevanceFeedbackService(
            store=JSONLFeedbackStore(_FEEDBACK_PATH),
        )
        self.lsi = LSIRetriever(
            repository=self.repository,
            document_store=self.document_store,
            model_dir=_MODELS_DIR,
            feedback_service=self.feedback_service,
        )
        _internet = InternetSearchRetriever(
            fetcher=WebContentFetcher(),
            document_store=self.document_store,
        )
        self.retriever = FallbackRetriever(
            primary=self.lsi,
            fallback=_internet,
            min_results=3,
            min_score=0.35,  # trigger web fallback when LSI top result scores below 35%
        )
        self.context = RetrievalContext(strategy=self.retriever)
        # The hybrid re-ranker (BM25 + LSI) is a first-class pipeline stage: it
        # runs between the post_retrieval and post_ranking hooks (see retrieve()),
        # exactly as the microkernel flow documents. RAGService therefore receives
        # already-ranked documents and does not re-rank them itself.
        self.ranker = HybridRanker()
        self.rag_service = RAGService()
        self.rag_evaluator = RAGEvaluator()
        # Content-based recommender (§4.2.3 — optional). Lives on its own
        # endpoint and feeds a separate UI panel, so it stays orthogonal
        # to the LSI ranking, Rocchio, and the evaluation dataset.
        self.recommender = ContentBasedRecommender(
            repository=self.repository,
            document_store=self.document_store,
        )
        # Microkernel: plugins are registered in build() once the vocabulary
        # the expansion plugin needs is available. Empty here = no-op pipeline.
        self.plugins = PluginPipeline()
        self.corpus = None
        self._source_label = ""

    def _model_exists(self) -> bool:
        """Return True if persisted LSI artifacts are present on disk."""
        path = Path(_MODELS_DIR)
        return (path / "tfidf.joblib").exists() and (path / "svd.joblib").exists()

    def build(self) -> None:
        """Load documents, build index, and fit the LSI model.

        On subsequent startups, loads persisted model artifacts from disk
        instead of re-parsing the full corpus (warm start).
        """
        print("  loading NLP model (spaCy)...", end=" ", flush=True)
        self.text_processor = TextProcessor()
        chunker = TextChunker(
            chunk_size=300,
            overlap=50,
            strategy="fixed",
        )
        self.indexer = IndexerService(text_processor=self.text_processor, chunker=chunker)
        print("done")

        if self._model_exists():
            try:
                self._warm_start()
                return
            except Exception as exc:
                # A saved model that fails to load (corrupt/incompatible artifacts,
                # renamed files, etc.) must not crash startup — fall back to a clean
                # cold start that rebuilds everything from the corpus.
                print(f"failed ({exc}); falling back to cold start", file=sys.stderr)

        self._cold_start()

    def _warm_start(self) -> None:
        """Load pre-built LSI model from disk — skips corpus parsing and SVD."""
        print("  found saved model — loading from disk (warm start)...", end=" ", flush=True)
        self.lsi = LSIRetriever.load(
            repository=self.repository,
            document_store=self.document_store,
            model_dir=_MODELS_DIR,
            feedback_service=self.feedback_service,
        )
        self.retriever.primary = self.lsi

        # Restore spell checker vocabulary so query correction works. The fitted
        # vocabulary is already persisted inside tfidf.joblib and restored by
        # LSIRetriever.load(), so we read it straight from the loaded processor —
        # no separate vocab artifact needed.
        for term in self.lsi.tfidf.vocabulary:
            self.text_processor.spell_checker._insert(term)

        n_docs = self.repository.collection.count()
        print(f"done  [{n_docs} docs in vector DB]")

        # Dynamic indexing: fold in any documents added to data/raw/ since the
        # last full fit, without re-fitting the SVD (Conf_2 "Incremental" path).
        self._fold_in_new_documents()

        # Plugins depend on the fitted vocabulary, now available after load().
        self._register_plugins()

    def _cold_start(self) -> None:
        """Parse corpus, fit LSI, persist artifacts for future warm starts."""
        print("  reading documents from data/raw/...", end=" ", flush=True)
        documents = _load_from_raw_dir()
        if not documents:
            raise RuntimeError(
                "No documents found in data/raw/. The pipeline cannot start "
                "without a corpus. Drop JSON/JSONL files exported by the "
                "crawler (or by scripts/ingest_pdfs.py) into data/raw/ and "
                "try again."
            )
        self._source_label = f"data/raw/ ({len(documents)} docs)"
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

        print("  saving model to disk...", end=" ", flush=True)
        Path(_MODELS_DIR).mkdir(parents=True, exist_ok=True)
        self.lsi.save(_MODELS_DIR)
        print("done  (future startups will be fast)")

        # Register optional plugins now that the fitted vocabulary is available.
        self._register_plugins()

    def _register_plugins(self) -> None:
        """Register optional pipeline plugins once the fitted vocabulary exists.

        Called from both cold and warm start so query expansion is always active,
        not just on the first ever run. The expansion plugin needs the fitted
        vocabulary to filter expanded terms to ones the TF-IDF model knows.

        Idempotent: the plugin pipeline is rebuilt from scratch on every call, so
        a warm start that also rebalances (which cold-starts internally) does not
        end up registering the same plugin twice.
        """
        self.plugins = PluginPipeline()
        self.plugins.register(
            QueryExpansionPlugin(target_vocabulary=self.lsi.tfidf.vocabulary)
        )

    def _fold_in_new_documents(self) -> None:
        """Incrementally index documents added to data/raw/ since the last fit.

        Detects raw documents whose source id is not yet represented in the
        document store, folds them into the existing latent space via
        :meth:`LSIRetriever.add_documents` (no SVD re-fit), and triggers a full
        rebuild when too many documents have been folded in (the "balanceo" step).
        A no-op when data/raw/ has no new documents.
        """
        raw_docs = _load_from_raw_dir()
        if not raw_docs:
            return

        # Stored ids are chunk ids ("<original>__chunk_<i>"); map back to the
        # original document id to compare against the raw corpus.
        indexed_originals = {
            stored_id.split("__chunk_")[0]
            for stored_id in self.document_store.list_all_ids()
        }
        new_docs = [d for d in raw_docs if d.doc_id not in indexed_originals]
        if not new_docs:
            return

        print(f"  folding in {len(new_docs)} new document(s)...", end=" ", flush=True)
        new_corpus = self.indexer.build(new_docs)
        added = self.lsi.add_documents(new_corpus)
        print(f"done  [+{added} chunks]")

        if self.lsi.needs_rebalance():
            print(
                f"  incremental fraction {self.lsi.incremental_fraction:.0%} exceeds "
                "threshold — rebalancing (full refit)..."
            )
            self._cold_start()

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        user_profile: UserProfile | None = None,
        force_web: bool = False,
    ) -> PipelineResult:
        """Run the full query pipeline and return all stage outputs.

        Returns:
            :class:`PipelineResult` bundling the retrieved documents, the
            RAG response (``None`` on failure), the terms added by the
            expansion plugin, and the RAG quality scores.
        """
        query_corpus = self.indexer.build_query(query_text)
        query = Query(text=query_text, indexed_corpus=query_corpus, user_profile=user_profile)

        # Microkernel flow (each hook is a no-op when no plugin is registered):
        #   pre_retrieval → retriever → post_retrieval → ranker → post_ranking → RAG
        context = PipelineContext(query=query)

        # pre_retrieval: plugins (e.g. query expansion) may rewrite the query's
        # indexed_corpus before the retriever runs.
        context = self.plugins.run_hook("pre_retrieval", context)
        query = context.query

        # Honour the UI toggle: temporarily lower min_results so the fallback
        # fires regardless of how many local results the LSI returned.
        original_min = self.retriever.min_results
        if force_web:
            self.retriever.min_results = top_k + 1

        # Retrieval.
        context.results = self.context.execute_search(query, top_k=top_k)

        if force_web:
            self.retriever.min_results = original_min

        # post_retrieval: plugins act on the raw retrieved set before ranking.
        context = self.plugins.run_hook("post_retrieval", context)

        # Ranking is a first-class stage: re-rank (BM25 + LSI) before generation.
        if context.results:
            context.results = self.ranker.rerank(query, context.results)

        # post_ranking: plugins act on the final ordering before answer generation.
        context = self.plugins.run_hook("post_ranking", context)
        results = context.results

        rag_response = None
        rag_quality: dict[str, float] | None = None
        if results:
            try:
                rag_response = self.rag_service.generate(query, results)
            except Exception as exc:
                print(f"  [warn] RAG generation failed: {exc}", file=sys.stderr)

            if rag_response is not None and getattr(rag_response, "answer", ""):
                try:
                    rag_quality = self.rag_evaluator.evaluate(
                        query, rag_response.answer, results,
                    )
                except Exception as exc:
                    print(f"  [warn] RAG evaluation failed: {exc}", file=sys.stderr)

        expansion = context.metadata.get("expansion", {})
        expansion_terms = list(expansion.get("added", []))

        return PipelineResult(
            results=results,
            rag_response=rag_response,
            expansion_terms=expansion_terms,
            rag_quality=rag_quality,
        )

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
        snippet = extract_snippet(r.document.text, query_text, length=200)
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


def _print_rag_quality(metrics: dict[str, float]) -> None:
    """Display the lightweight RAGAS-style quality metrics (Conf_9)."""
    parts = [f"{name}={value:.2f}" for name, value in metrics.items()]
    print(f"\n  ──── Calidad RAG ────  {'  '.join(parts)}")


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

            outcome = pipeline.retrieve(raw, user_profile=user_profile)
            _print_results(outcome.results, raw)
            if outcome.expansion_terms:
                print(f"\n  (también buscamos: {', '.join(outcome.expansion_terms)})")
            if outcome.rag_response:
                _print_rag_response(outcome.rag_response)
            if outcome.rag_quality:
                _print_rag_quality(outcome.rag_quality)
            print()

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


def run_oneshot(pipeline: Pipeline, query: str, user_profile: UserProfile | None = None) -> None:
    outcome = pipeline.retrieve(query, user_profile=user_profile)
    _print_results(outcome.results, query)
    if outcome.expansion_terms:
        print(f"\n  (también buscamos: {', '.join(outcome.expansion_terms)})")
    if outcome.rag_response:
        _print_rag_response(outcome.rag_response)
    if outcome.rag_quality:
        _print_rag_quality(outcome.rag_quality)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShealtRI — Medical Information Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", help="Run a single query and exit")
    parser.add_argument("--stats", action="store_true", help="Print corpus stats and exit")
    parser.add_argument(
        "--eval", action="store_true",
        help="Evaluate the LSI retriever against the bundled test collection "
             "(P@k, R@k, F1, NDCG, MAP, MRR) and exit",
    )
    parser.add_argument(
        "--eval-k", type=int, default=10, metavar="K",
        help="Cut-off rank for the @k evaluation metrics (default: 10)",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="With --eval, apply the HybridRanker (BM25+LSI) before scoring "
             "to compare against pure LSI",
    )
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

    if args.eval:
        _run_evaluation(pipeline, k=args.eval_k, rerank=args.rerank)
    elif args.stats:
        _print_stats(pipeline.stats())
    elif args.query:
        run_oneshot(pipeline, args.query, user_profile=user_profile)
    else:
        run_interactive(pipeline, user_profile=user_profile)


def _run_evaluation(pipeline: Pipeline, k: int, rerank: bool) -> None:
    """Evaluate the pure LSI retriever against the bundled test collection.

    Reuses the evaluation module's machinery so the CLI metrics match
    ``python -m modules.evaluation.service``. The pipeline's own HybridRanker is
    passed through when ``rerank`` is set, so the comparison reflects the live
    ranking stage rather than a freshly constructed one.
    """
    from modules.evaluation.dataset import load_dataset
    from modules.evaluation.service import EvaluationService, _build_lsi_search_fn

    print("[eval] loading bundled test collection...")
    dataset = load_dataset(
        "data/evaluation/eval_queries.json",
        "data/evaluation/eval_qrels.json",
    )
    ranker = pipeline.ranker if rerank else None
    if rerank:
        print("[eval] re-ranking enabled (HybridRanker BM25+LSI)")

    print(f"[eval] running evaluation (k={k})...\n")
    service = EvaluationService(_build_lsi_search_fn(pipeline, ranker=ranker), k=k)
    report = service.evaluate(dataset)
    print(report.format_table())


if __name__ == "__main__":
    main()
