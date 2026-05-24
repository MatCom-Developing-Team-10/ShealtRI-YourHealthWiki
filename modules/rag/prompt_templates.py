"""Prompt rendering for profile-aware RAG.

Keeps all string construction in one place so the service stays clean.
Uses str.format_map — no Jinja2 dependency needed.
"""

from __future__ import annotations

from core.models import RetrievedDocument
from .user_profiles import ProfileConfig


def build_context_block(docs: list[RetrievedDocument], max_docs: int = 3) -> str:
    """Build the context section from retrieved documents.

    Each document is presented as a numbered block with its title/URL
    and a snippet of text. Long texts are truncated to avoid overflow.

    Args:
        docs: Retrieved documents, already ranked by score.
        max_docs: Maximum number of documents to include.

    Returns:
        Multi-line string ready for insertion into the system prompt.
    """
    selected = docs[:max_docs]
    if not selected:
        return "No se encontraron documentos de referencia relevantes."

    lines: list[str] = []
    for i, rd in enumerate(selected, start=1):
        title = rd.document.metadata.get("title", rd.document.doc_id)
        url = rd.document.url or "(sin URL)"
        snippet = rd.document.text[:800].replace("\n", " ").strip()
        if len(rd.document.text) > 800:
            snippet += "..."
        lines.append(
            f"[Documento {i}] {title}\nFuente: {url}\n{snippet}"
        )

    return "\n\n---\n\n".join(lines)


def render_prompt(
    profile_config: ProfileConfig,
    query_text: str,
    context_block: str,
) -> str:
    """Render the final prompt string using the profile's system_prompt template.

    Args:
        profile_config: The active profile configuration.
        query_text: Raw query string from the user.
        context_block: Pre-built context from build_context_block().

    Returns:
        Complete prompt string to send to the LLM.
    """
    return profile_config.system_prompt.format_map({
        "query": query_text,
        "context": context_block,
        "focus_areas": ", ".join(profile_config.focus_areas),
    })


def build_fallback_response(
    profile_config: ProfileConfig,
    query_text: str,
    docs: list[RetrievedDocument],
    max_docs: int = 3,
) -> str:
    """Build a template-based response when the LLM is unavailable.

    The fallback presents the retrieved document snippets directly, framed
    by a short introduction appropriate for the profile's tone.

    Args:
        profile_config: Active profile — used for the framing sentence.
        query_text: Original user query.
        docs: Retrieved documents.
        max_docs: Maximum documents to present.

    Returns:
        Plain-text response string in Spanish.
    """
    selected = docs[:max_docs]

    intro_by_profile = {
        "paciente": (
            "Aquí tienes información relevante sobre tu consulta. "
            "Recuerda consultar a tu médico para una evaluación personalizada."
        ),
        "estudiante_medicina": (
            "A continuación, los documentos de referencia más relevantes "
            "para tu consulta académica:"
        ),
        "profesional_medico": (
            "Documentos de referencia clínica para la consulta planteada:"
        ),
        "diagnostico_asistido": (
            "Información de referencia para apoyo diagnóstico. "
            "Evalúe en contexto clínico:"
        ),
        "medicina_natural": (
            "Información de medicina integrativa y tradicional disponible:"
        ),
        "cuidador_familiar": (
            "Información práctica para apoyar el cuidado en casa:"
        ),
    }

    intro = intro_by_profile.get(
        profile_config.profile_type.value,
        "Información relevante encontrada:",
    )

    if not selected:
        return (
            f"{intro}\n\n"
            f"No se encontraron documentos relevantes para la consulta: «{query_text}»."
        )

    parts = [f"{intro}\n"]
    for i, rd in enumerate(selected, start=1):
        title = rd.document.metadata.get("title", rd.document.doc_id)
        url = rd.document.url or "(sin URL)"
        snippet = rd.document.text[:600].replace("\n", " ").strip()
        if len(rd.document.text) > 600:
            snippet += "..."
        parts.append(f"{i}. **{title}**\n   Fuente: {url}\n   {snippet}")

    return "\n\n".join(parts)
