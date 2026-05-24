"""Spanish medical thesaurus for query expansion.

Maps each canonical (lemmatised) medical term to a frozen set of related
terms — synonyms, abbreviations, and lay/technical equivalents — that
should be considered for query expansion.

Design notes
------------
* All keys and values are in **lower-case, lemmatised** form so they match
  what ``modules.text_processor.TextProcessor.process()`` emits after
  normalisation. Tests in ``tests/regression/test_thesaurus_lockdown.py``
  pin the most important entries against regressions.

* The map is **bidirectional**: if ``"hipertensión"`` maps to ``"hta"``,
  ``"hta"`` also maps to ``"hipertensión"``. This is enforced by
  ``_build_bidirectional()`` at import time so the source data below
  remains compact.

* The thesaurus is intentionally small (~30 keys, ~60 unique terms). It
  covers the 20 topics in the synthetic test corpus plus the medical
  abbreviations already declared in
  ``modules.text_processor.stopwords.MEDICAL_ABBREVIATIONS``. Extending it
  is a one-line operation; updating tests is *not* — please add entries
  here only when they show up in real queries that fail without expansion.
"""

from __future__ import annotations

from typing import FrozenSet

# Source data: canonical term -> set of equivalents.
# Bidirectionality is added programmatically below.
#
# Lemma-form notes
# ----------------
# Some Spanish words are lemmatised in unexpected ways by spaCy
# (es_core_news_md 3.8). When the corpus stores the lemmatised form but
# the human-readable synonym differs, we register BOTH forms here so the
# target_vocabulary filter in :class:`plugins.expansion.QueryExpander` has
# a chance to keep something:
#
#     glucosa     → spaCy emits ``glucós``
#     hemoglobina → spaCy emits ``hemoglobinar``
#     ánimo       → spaCy emits ``animar``
#
# The regression test ``tests/regression/test_thesaurus_lockdown.py``
# guards the edges that depend on this; if spaCy changes its lemmatiser
# behaviour, the test will surface the breakage.
_RAW_THESAURUS: dict[str, set[str]] = {
    # ---- Cardiovascular ----
    "hipertensión": {"hta", "presión", "tensión", "arterial"},
    "infarto": {"iam", "miocardio", "ataque", "cardiovascular"},
    "ictus": {"acv", "cerebrovascular", "trombolisis"},
    "miocardio": {"infarto", "iam", "corazón", "cardíaco"},
    "corazón": {"cardíaco", "cardiovascular", "miocardio"},
    # ---- Endocrine ----
    # 'glucós' is the spaCy lemma of 'glucosa'; both are listed.
    "diabetes": {"dm", "glucemia", "glucosa", "glucós", "insulina", "hiperglucemia"},
    "glucosa": {"glucemia", "glucós", "diabetes", "azúcar"},
    "hipotiroidismo": {"tiroides", "hormona", "tsh", "levotiroxina"},
    "tiroides": {"hipotiroidismo", "hormona", "tsh"},
    # ---- Respiratory ----
    "asma": {"bronquial", "respiratorio", "disnea", "broncodilatador"},
    "epoc": {"enfisema", "bronquitis", "pulmonar", "tabaquismo"},
    "neumonía": {"pulmón", "pulmonar", "infección", "respiratorio"},
    # ---- Renal ----
    "renal": {"riñón", "diálisis", "creatinina", "nefropatía"},
    "riñón": {"renal", "diálisis"},
    # ---- Mental health / neuro ----
    # 'animar' is the spaCy lemma of 'ánimo'.
    "depresión": {"trastorno", "ánimo", "animar", "antidepresivo"},
    "alzheimer": {"demencia", "neurodegenerativa", "memoria", "cognitivo"},
    "demencia": {"alzheimer", "memoria", "cognitivo"},
    "migraña": {"cefalea", "dolor", "triptán", "aura"},
    "cefalea": {"migraña", "dolor"},
    # ---- Musculoskeletal ----
    "osteoporosis": {"hueso", "densidad", "fractura", "calcio", "bifosfonato"},
    "artritis": {"articulación", "inflamación", "reumatoide", "autoinmune"},
    "fibromialgia": {"dolor", "musculoesquelético", "fatiga", "crónico"},
    # ---- Oncology / gastro / liver ----
    "cáncer": {"tumor", "oncológico", "maligno", "quimioterapia"},
    "colon": {"colorrectal", "intestino", "colonoscopia"},
    "hepatitis": {"hígado", "hepático", "viral", "cirrosis"},
    "hígado": {"hepático", "hepatitis", "cirrosis"},
    # ---- Metabolic / lipid ----
    "colesterol": {"ldl", "hdl", "lípido", "estatina", "atorvastatina"},
    "obesidad": {"imc", "metabólico", "sobrepeso", "bariátrica"},
    # ---- Haematology ----
    # 'hemoglobinar' is the spaCy lemma of 'hemoglobina'.
    "anemia": {"hemoglobina", "hemoglobinar", "hierro", "ferritina", "ferropénica"},
    "hemoglobina": {"anemia", "hemoglobinar", "hierro", "hba1c"},
    # ---- Lay-to-technical bridges (one-way is enough — the loop below
    #      makes them bidirectional automatically) ----
    "ataque": {"infarto", "iam"},
    # 'azúcar' must also reach the broken 'glucós' lemma so it survives
    # the target_vocabulary filter on real corpora.
    "azúcar": {"glucosa", "glucós", "glucemia"},
    "presión": {"hipertensión", "hta", "tensión"},
    "tensión": {"hipertensión", "presión", "hta"},
}


def _build_bidirectional(raw: dict[str, set[str]]) -> dict[str, FrozenSet[str]]:
    """Make every (a -> b) edge in ``raw`` symmetric: b also maps back to a.

    Returns frozenset values so callers cannot accidentally mutate the
    thesaurus at runtime.
    """
    merged: dict[str, set[str]] = {k: set(v) for k, v in raw.items()}
    for term, synonyms in raw.items():
        for syn in synonyms:
            merged.setdefault(syn, set()).add(term)
    # Strip self-loops just in case
    for term, synonyms in merged.items():
        synonyms.discard(term)
    return {k: frozenset(v) for k, v in merged.items()}


#: Final read-only thesaurus consumed by :func:`expand_term`.
MEDICAL_THESAURUS: dict[str, FrozenSet[str]] = _build_bidirectional(_RAW_THESAURUS)


def expand_term(term: str) -> FrozenSet[str]:
    """Return the set of synonyms registered for ``term``, or empty.

    Args:
        term: A lower-case, lemmatised term to look up.

    Returns:
        Frozen set of related terms (excluding the term itself). Empty
        frozenset if the term is not in the thesaurus — callers can use the
        result directly in set operations without a None check.
    """
    return MEDICAL_THESAURUS.get(term.lower(), frozenset())
