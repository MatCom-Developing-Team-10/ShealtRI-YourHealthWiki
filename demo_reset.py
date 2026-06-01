"""demo_reset.py — ShealtRI demo: wipe all derived data before a live demo.

    python demo_reset.py          # interactive confirmation
    python demo_reset.py --yes    # skip confirmation (non-interactive)

Removes:
  - data/chroma/     (ChromaDB vector database)
  - data/documents/  (document store — indexed chunks as JSON)
  - models/lsi/      (trained TF-IDF + SVD artifacts)

Preserves:
  - data/raw/           (PDF books + crawler JSONL output — source corpus)
  - data/evaluation/    (evaluation queries and qrels)

After this script completes, run:
    python cli.py --stats
to re-index the corpus from scratch and train the LSI model.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

_TARGETS = [
    _ROOT / "data" / "chroma",
    _ROOT / "data" / "documents",
    _ROOT / "models" / "lsi",
]

_PRESERVE = [
    _ROOT / "data" / "raw",
    _ROOT / "data" / "evaluation",
]


def _bytes_to_human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShealtRI demo reset — wipe derived data for a clean demo",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║         ShealtRI — Demo Reset                                ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Directories that will be DELETED:")
    print()

    total_size = 0
    for target in _TARGETS:
        size = _dir_size(target)
        total_size += size
        exists = "✓ exists" if target.exists() else "  (already empty)"
        print(f"    ✗  {target.relative_to(_ROOT)}    [{_bytes_to_human(size)}]  {exists}")

    print()
    print(f"  Total to free: {_bytes_to_human(total_size)}")
    print()
    print("  Directories that will be PRESERVED:")
    for preserved in _PRESERVE:
        print(f"    ✓  {preserved.relative_to(_ROOT)}")
    print()

    if not args.yes:
        try:
            answer = input("  ¿Continuar? [s/N]  ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelado.")
            sys.exit(0)

        if answer not in {"s", "si", "sí", "y", "yes"}:
            print("  Cancelado.")
            sys.exit(0)

    # data/chroma is intentionally NOT recreated: ChromaDB creates it fresh on
    # first access, which avoids the SQLITE_READONLY_DBMOVED (1032) error that
    # occurs when an empty directory is pre-created and ChromaDB finds stale WAL
    # state from a previous process.
    # data/documents and models/lsi ARE pre-created because FileSystemDocumentStore
    # and joblib.dump expect the parent directory to exist at write time.
    _NO_RECREATE = {_ROOT / "data" / "chroma"}

    print()
    for target in _TARGETS:
        if not target.exists():
            print(f"  skip  {target.relative_to(_ROOT)}  (no existe)")
            continue
        try:
            shutil.rmtree(target)
            if target not in _NO_RECREATE:
                target.mkdir(parents=True, exist_ok=True)
                print(f"  ✓  {target.relative_to(_ROOT)}  eliminado y recreado vacío")
            else:
                print(f"  ✓  {target.relative_to(_ROOT)}  eliminado  (ChromaDB lo creará al arrancar)")
        except OSError as exc:
            print(f"  ✗  {target.relative_to(_ROOT)}  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    print()
    print("  ══════════════════════════════════════════════════════════════")
    print("  ✓  Reset completo. Los datos están en estado inicial.")
    print()
    print("  Para indexar el corpus y entrenar el modelo LSI, ejecute:")
    print()
    print("      python cli.py --stats")
    print()
    print("  (Primera ejecución con corpus completo: ~8-10 minutos)")
    print()


if __name__ == "__main__":
    main()
