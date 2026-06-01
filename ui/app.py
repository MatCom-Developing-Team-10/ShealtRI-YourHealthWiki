"""FastAPI backend for ShealtRI web UI.

Serves:
  - HTML/CSS/JS single-page app from ui/static/
  - API endpoints for query execution and profile management
  - MP4 video files for avatar states
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.models import UserProfile, UserProfileType
from cli import Pipeline
from modules.document_reconstructor import DocumentReconstructor
from modules.evaluation.service import EvaluationService, _build_lsi_search_fn
from modules.evaluation.dataset import load_dataset

_document_reconstructor = DocumentReconstructor(
    documents_dir=_PROJECT_ROOT / "data" / "documents",
    cache_dir=_PROJECT_ROOT / "data" / "pdf_reconstructed",
)


def extract_snippet(text: str, query: str, length: int = 300) -> str:
    """Return a query-aware excerpt from text.

    Slides a window over the text and picks the position where the most
    query terms appear, then returns a fragment of `length` characters
    centred on that position.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models for API requests/responses
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    profile: str = "paciente"
    top_k: int = 5
    force_web: bool = False
    apply_correction: bool = True


class DocumentResult(BaseModel):
    doc_id: str
    title: str
    source: str
    snippet: str
    relevance: float
    source_type: str = "local"  # "local" or "web"


class QueryResponse(BaseModel):
    results: list[DocumentResult]
    rag_response: str
    query_time_ms: int
    corrected_query: str | None = None
    correction_applied: bool = False
    used_web_fallback: bool = False
    expansion_terms: list[str] = []
    rag_quality: dict[str, float] | None = None
    rag_model_name: str | None = None
    rag_used_llm: bool = False


class FeedbackRequest(BaseModel):
    query: str
    doc_id: str
    relevant: bool


class FeedbackResponse(BaseModel):
    recorded: bool
    total_judgments_for_query: int


class PerQueryEvalResult(BaseModel):
    query_id: str
    text: str
    num_relevant: int
    num_retrieved: int
    metrics: dict[str, float]


class EvalResponse(BaseModel):
    aggregated: dict[str, float]
    k: int
    num_queries: int
    corpus_size: int
    per_query: list[PerQueryEvalResult]


class RecommendRequest(BaseModel):
    seed_doc_ids: list[str]
    top_k: int = 3
    profile: str = "paciente"
    exclude_ids: list[str] = []


class RecommendedItem(BaseModel):
    doc_id: str
    title: str
    source: str
    snippet: str
    similarity: float
    source_type: str = "local"


class RecommendResponse(BaseModel):
    items: list[RecommendedItem]


class StatsResponse(BaseModel):
    n_documents: int
    n_chunks: int
    n_terms: int
    avg_tokens_per_doc: float
    avg_postings_per_term: float
    lsi_k: int
    feedback_judgments: int


class ProfileOption(BaseModel):
    slug: str
    label: str


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app and pipeline setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="ShealtRI UI Backend", version="1.0.0")

# Load and build the pipeline once on startup
_pipeline = Pipeline()
print("[startup] Loading pipeline...", flush=True)
_pipeline.build()
print("[startup] Pipeline ready.", flush=True)

# Profile mapping
_PROFILE_MAP: dict[str, UserProfileType] = {
    "paciente": UserProfileType.PATIENT,
    "estudiante": UserProfileType.MEDICAL_STUDENT,
    "medico": UserProfileType.MEDICAL_PROFESSIONAL,
    "diagnostico": UserProfileType.DIAGNOSTIC_ASSISTANT,
    "natural": UserProfileType.NATURAL_MEDICINE,
    "cuidador": UserProfileType.CAREGIVER,
}

_PROFILES = [
    ProfileOption(slug="paciente", label="Paciente"),
    ProfileOption(slug="estudiante", label="Estudiante de medicina"),
    ProfileOption(slug="medico", label="Profesional médico"),
    ProfileOption(slug="diagnostico", label="Diagnóstico médico"),
    ProfileOption(slug="natural", label="Medicina natural o tradicional"),
    ProfileOption(slug="cuidador", label="Cuidador/Familiar"),
]


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/profiles")
def get_profiles() -> list[ProfileOption]:
    """Return list of available user profiles."""
    return _PROFILES


@app.post("/api/query")
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Execute a query and return results + RAG response."""
    start_time = time.time()

    # Build user profile
    profile_type = _PROFILE_MAP.get(req.profile, UserProfileType.PATIENT)
    user_profile = UserProfile(
        profile_type=profile_type,
        name=req.profile.capitalize(),
    )

    # Run pipeline. ``apply_correction`` honours the UI choice: when the user
    # opted to search exactly what they typed, the query is not spell-corrected.
    outcome = _pipeline.retrieve(
        query_text=req.query,
        top_k=req.top_k,
        user_profile=user_profile,
        force_web=req.force_web,
        apply_correction=req.apply_correction,
    )

    # Convert results to JSON-serializable format
    doc_results = []
    for result in (outcome.results or []):
        title = result.document.metadata.get("title", result.document.doc_id)
        snippet = extract_snippet(result.document.text, req.query, length=300)

        source_type = result.document.metadata.get("origin", "local")
        doc_results.append(DocumentResult(
            doc_id=result.document.doc_id,
            title=title,
            source=result.document.url or "(sin URL)",
            snippet=snippet,
            relevance=float(result.score),
            source_type=source_type,
        ))

    used_web_fallback = any(r.source_type == "web" for r in doc_results)

    # Detect the *available* spell correction independently of whether it was
    # applied to this search. We always force correction here (no side effects
    # post-fit) so the UI can offer the user a choice in both directions:
    #   - correction applied  → "search instead for what I typed"
    #   - correction skipped   → "did you mean <correction>?"
    corrected_query: str | None = None
    try:
        query_corpus = _pipeline.indexer.build_query(req.query, apply_correction=True)
        corrected = query_corpus.corrected_text
        if corrected and corrected.strip() != req.query.strip().lower():
            corrected_query = corrected
    except Exception:
        pass

    # Whether the retrieval that actually ran used the corrected query.
    correction_applied = bool(req.apply_correction and corrected_query)

    # Extract RAG response text + provenance
    rag_text = ""
    rag_model_name: str | None = None
    rag_used_llm = False
    rag_response = outcome.rag_response
    if rag_response is not None:
        rag_text = getattr(rag_response, "answer", "") or ""
        rag_model_name = getattr(rag_response, "model_name", None)
        rag_used_llm = bool(getattr(rag_response, "used_llm", False))

    elapsed_ms = int((time.time() - start_time) * 1000)

    return QueryResponse(
        results=doc_results,
        rag_response=rag_text,
        query_time_ms=elapsed_ms,
        corrected_query=corrected_query,
        correction_applied=correction_applied,
        used_web_fallback=used_web_fallback,
        expansion_terms=outcome.expansion_terms,
        rag_quality=outcome.rag_quality,
        rag_model_name=rag_model_name,
        rag_used_llm=rag_used_llm,
    )


@app.post("/api/feedback")
def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
    """Record one relevance judgment so Rocchio re-weights the next query.

    The next time the same query text is searched, ``LSIRetriever`` will
    consult ``feedback_service`` and pull the latent query vector toward
    the centroid of docs marked relevant and away from those marked
    non-relevant.
    """
    _pipeline.feedback_service.record(
        query_text=req.query, doc_id=req.doc_id, relevant=req.relevant,
    )
    rel, non_rel = _pipeline.feedback_service.store.get_for_query(req.query)
    return FeedbackResponse(
        recorded=True, total_judgments_for_query=len(rel) + len(non_rel),
    )


@app.post("/api/recommend")
def recommend_endpoint(req: RecommendRequest) -> RecommendResponse:
    """Return content-based recommendations seeded by the latest result set.

    Lives on its own endpoint so it stays orthogonal to ``/api/query``:
    Rocchio re-weighting, HybridRanker, and the evaluation dataset are
    unaffected. The profile-based source boost only nudges ties — the
    recommender remains similarity-first (Conf_6 + Conf_11).
    """
    profile_type = _PROFILE_MAP.get(req.profile, UserProfileType.PATIENT)
    recs = _pipeline.recommender.recommend(
        seed_doc_ids=req.seed_doc_ids,
        top_k=req.top_k,
        exclude_ids=set(req.exclude_ids),
        profile=profile_type,
    )

    items: list[RecommendedItem] = []
    for r in recs:
        doc = r.document
        title = doc.metadata.get("title", doc.doc_id)
        snippet = extract_snippet(doc.text, " ".join(doc.metadata.get("title", "").split()), length=180)
        items.append(RecommendedItem(
            doc_id=doc.doc_id,
            title=title,
            source=doc.url or "(sin URL)",
            snippet=snippet,
            similarity=float(r.similarity),
            source_type=doc.metadata.get("origin", "local"),
        ))
    return RecommendResponse(items=items)


@app.get("/api/document")
def serve_document(path: str) -> FileResponse:
    """Serve a local document file, reconstructing from chunks if missing.

    When the original PDF cannot be found on disk, the request is satisfied
    by asking ``DocumentReconstructor`` to rebuild a text-only PDF from every
    indexed chunk that shares the same ``url``. The reconstruction is cached
    under ``data/pdf_reconstructed`` so the linear scan only runs once per
    missing source.
    """
    resolved = (_PROJECT_ROOT / path).resolve()
    allowed_root = (_PROJECT_ROOT / "data").resolve()
    if not str(resolved).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Access denied")
    if resolved.exists():
        return FileResponse(str(resolved))

    reconstructed = _document_reconstructor.reconstruct(path)
    if reconstructed is not None and reconstructed.exists():
        return FileResponse(
            str(reconstructed),
            media_type="application/pdf",
            filename=reconstructed.name,
        )
    raise HTTPException(status_code=404, detail="File not found")


def _original_doc_count() -> int:
    """Count distinct original documents in the indexed corpus.

    The LSI index is built over chunks; ``_build_lsi_search_fn`` collapses
    chunk results back to their parent document id, so Fallout must be
    computed against the same denominator (number of *original* docs).
    """
    if _pipeline.corpus is None:
        return 0
    originals: set[str] = set()
    for doc in _pipeline.corpus.documents:
        original = doc.metadata.get("original_doc_id") or doc.doc_id.split("__chunk_")[0]
        originals.add(original)
    return len(originals)


@app.get("/api/eval")
def eval_endpoint(k: int = 10) -> EvalResponse:
    """Run the bundled evaluation dataset against the LSI retriever."""
    data_dir = _PROJECT_ROOT / "data" / "evaluation"
    dataset = load_dataset(
        str(data_dir / "eval_queries.json"),
        str(data_dir / "eval_qrels.json"),
    )
    search_fn = _build_lsi_search_fn(_pipeline)
    corpus_size = _original_doc_count()
    report = EvaluationService(
        search_fn, k=k, corpus_size=corpus_size,
    ).evaluate(dataset)

    per_query = [
        PerQueryEvalResult(
            query_id=r.query_id,
            text=r.text,
            num_relevant=r.num_relevant,
            num_retrieved=r.num_retrieved,
            metrics=r.metrics,
        )
        for r in report.per_query
    ]
    return EvalResponse(
        aggregated=report.aggregated,
        k=report.k,
        num_queries=len(report.per_query),
        corpus_size=corpus_size,
        per_query=per_query,
    )


@app.get("/api/stats")
def stats_endpoint() -> StatsResponse:
    """Corpus / model diagnostics for the UI footer."""
    raw = _pipeline.stats()
    lsi_k = 0
    if _pipeline.lsi is not None and _pipeline.lsi.svd is not None:
        lsi_k = int(_pipeline.lsi.svd.n_components)

    # Total feedback judgments recorded so far across all queries.
    judgments = len(_pipeline.feedback_service.store)

    return StatsResponse(
        n_documents=_original_doc_count(),
        n_chunks=int(raw.get("n_documents", 0)),
        n_terms=int(raw.get("n_terms", 0)),
        avg_tokens_per_doc=float(raw.get("avg_tokens_per_doc", 0.0)),
        avg_postings_per_term=float(raw.get("avg_postings_per_term", 0.0)),
        lsi_k=lsi_k,
        feedback_judgments=judgments,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Static files
# ─────────────────────────────────────────────────────────────────────────────

# Mount static files (CSS, JS, videos)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root() -> FileResponse:
    """Serve the main SPA."""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/videos/{filename}")
async def get_video(filename: str) -> FileResponse:
    """Serve video files for avatar states."""
    video_path = static_dir / "videos" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path=video_path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8501,
        log_level="info",
    )
