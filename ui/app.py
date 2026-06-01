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


class EvalResponse(BaseModel):
    aggregated: dict[str, float]
    k: int
    num_queries: int


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
    results, rag_response = _pipeline.retrieve(
        query_text=req.query,
        top_k=req.top_k,
        user_profile=user_profile,
        force_web=req.force_web,
        apply_correction=req.apply_correction,
    )

    # Convert results to JSON-serializable format
    doc_results = []
    for result in (results or []):
        title = result.document.metadata.get("title", result.document.doc_id)
        snippet = extract_snippet(result.document.text, req.query, length=300)

        source_type = result.document.metadata.get("origin", "local")
        doc_results.append(DocumentResult(
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

    # Extract RAG response text
    rag_text = ""
    if rag_response and hasattr(rag_response, "answer"):
        rag_text = rag_response.answer

    elapsed_ms = int((time.time() - start_time) * 1000)

    return QueryResponse(
        results=doc_results,
        rag_response=rag_text,
        query_time_ms=elapsed_ms,
        corrected_query=corrected_query,
        correction_applied=correction_applied,
        used_web_fallback=used_web_fallback,
    )


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


@app.get("/api/eval")
def eval_endpoint(k: int = 10) -> EvalResponse:
    """Run the bundled evaluation dataset against the LSI retriever."""
    data_dir = _PROJECT_ROOT / "data" / "evaluation"
    dataset = load_dataset(
        str(data_dir / "eval_queries.json"),
        str(data_dir / "eval_qrels.json"),
    )
    search_fn = _build_lsi_search_fn(_pipeline)
    report = EvaluationService(search_fn, k=k).evaluate(dataset)
    return EvalResponse(
        aggregated=report.aggregated,
        k=report.k,
        num_queries=len(report.per_query),
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
