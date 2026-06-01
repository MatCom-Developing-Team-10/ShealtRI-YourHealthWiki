"""Demo standalone de expansión de queries usando el tesauro médico.

No requiere cargar el pipeline completo ni el modelo LSI.
Muestra qué términos se agregarían a una query antes de la búsqueda.

Uso:
    python demo_expansion.py "hipertensión arterial"
    python demo_expansion.py "diabetes insulina"
    python demo_expansion.py "dolor de cabeza"
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.expansion.thesaurus import expand_term, MEDICAL_THESAURUS


def show_expansion(query: str, max_terms: int = 6) -> None:
    tokens = [t.lower().strip() for t in query.split() if t.strip()]

    print()
    print("=" * 60)
    print(f"  Query original : \"{query}\"")
    print(f"  Tokens         : {tokens}")
    print("=" * 60)

    original_set = set(tokens)
    all_candidates: set[str] = set()

    print("\n  Expansión por término:")
    for token in tokens:
        synonyms = expand_term(token)
        new_synonyms = synonyms - original_set
        if new_synonyms:
            print(f"    '{token}'  →  {sorted(new_synonyms)}")
        else:
            print(f"    '{token}'  →  (no hay sinónimos en el tesauro)")
        all_candidates.update(new_synonyms)

    # Remove original terms and cap at max_terms
    all_candidates -= original_set
    if len(all_candidates) > max_terms:
        all_candidates = set(sorted(all_candidates)[:max_terms])

    print()
    print(f"  Términos originales : {sorted(original_set)}")
    print(f"  Términos agregados  : {sorted(all_candidates)}")
    print(f"  Query expandida     : {sorted(original_set | all_candidates)}")
    print()

    if not all_candidates:
        print("  ⚠  Ningún término del tesauro coincide con esta query.")
        print("     En el pipeline real, la query pasaría sin cambios al retriever.")
    else:
        print(f"  ✓  Se agregarían {len(all_candidates)} término(s) antes de la búsqueda LSI.")
    print()


def list_thesaurus_entries() -> None:
    print("\n  Términos disponibles en el tesauro médico:")
    for term in sorted(MEDICAL_THESAURUS.keys()):
        print(f"    {term}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n  Ejemplos de queries con expansión:")
        examples = [
            "hipertensión arterial",
            "diabetes insulina",
            "infarto miocardio",
            "dolor cabeza",
            "anemia hemoglobina",
        ]
        for ex in examples:
            show_expansion(ex)
    elif sys.argv[1] == "--list":
        list_thesaurus_entries()
    else:
        query = " ".join(sys.argv[1:])
        show_expansion(query)
