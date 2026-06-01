"""demo_status.py — ShealtRI demo: show current state of all data stores.

    python demo_status.py

Prints a clean summary of:
  - LSI model artifacts (trained / not trained)
  - ChromaDB vector count
  - Document store size
  - data/raw/ contents (PDF books + JSONL crawler output)
  - Evaluation dataset availability
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _sep(title: str = "") -> None:
    if title:
        pad = (60 - len(title) - 2) // 2
        print(f"\n  {'─' * pad} {title} {'─' * pad}")
    else:
        print(f"\n  {'─' * 62}")


def check_lsi_model() -> dict:
    model_dir = _ROOT / "models" / "lsi"
    # tfidf.joblib and svd.joblib are the two artifacts written by TfidfProcessor.save()
    # and LSIModel.save(). vocab.joblib is NOT written by the pipeline.
    artifacts = ["tfidf.joblib", "svd.joblib"]
    found = {a: (model_dir / a).exists() for a in artifacts}
    return {"dir": model_dir, "artifacts": found, "ready": all(found.values())}


def check_chroma() -> dict:
    try:
        import chromadb  # noqa: PLC0415
        client = chromadb.PersistentClient(str(_ROOT / "data" / "chroma"))
        collection = client.get_collection("medical_documents")
        count = collection.count()
        return {"count": count, "error": None}
    except ModuleNotFoundError:
        return {
            "count": 0,
            "error": "chromadb not found — activate the virtualenv first:\n"
                     "    source .venv/bin/activate",
        }
    except Exception as e:  # noqa: BLE001
        # Collection doesn't exist yet = empty, not an error
        if "does not exist" in str(e) or "not found" in str(e).lower():
            return {"count": 0, "error": None}
        return {"count": 0, "error": str(e)}


def check_document_store() -> dict:
    store = _ROOT / "data" / "documents"
    if not store.exists():
        return {"count": 0, "exists": False}
    files = list(store.glob("*.json"))
    return {"count": len(files), "exists": True}


def check_raw_dir() -> dict:
    raw = _ROOT / "data" / "raw"
    if not raw.exists():
        return {"pdfs": 0, "jsonl": 0, "exists": False}
    pdfs = list(raw.glob("*.pdf"))
    jsonl = list(raw.glob("*.jsonl"))
    return {"pdfs": len(pdfs), "jsonl": len(jsonl), "exists": True, "path": raw,
            "needs_ingest": len(pdfs) > 0 and len(jsonl) == 0}


def check_eval_dataset() -> dict:
    queries = _ROOT / "data" / "evaluation" / "eval_queries.json"
    qrels = _ROOT / "data" / "evaluation" / "eval_qrels.json"
    return {
        "queries": queries.exists(),
        "qrels": qrels.exists(),
        "ready": queries.exists() and qrels.exists(),
    }


def main() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║         ShealtRI — Data State Inspector                      ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    # ── LSI model ──────────────────────────────────────────────────────────
    _sep("LSI MODEL")
    lsi = check_lsi_model()
    status = "✓  READY" if lsi["ready"] else "✗  NOT TRAINED"
    print(f"  Status    : {status}")
    for name, exists in lsi["artifacts"].items():
        mark = "✓" if exists else "✗"
        print(f"    {mark}  {lsi['dir'] / name}")

    # ── ChromaDB ───────────────────────────────────────────────────────────
    _sep("VECTOR DATABASE (ChromaDB)")
    chroma = check_chroma()
    if chroma["error"]:
        print(f"  Status    : ✗  ERROR — {chroma['error']}")
    else:
        status = "✓  INDEXED" if chroma["count"] > 0 else "✗  EMPTY"
        print(f"  Status    : {status}")
        print(f"  Vectors   : {chroma['count']:,}")

    # ── Document store ─────────────────────────────────────────────────────
    _sep("DOCUMENT STORE")
    store = check_document_store()
    if not store["exists"]:
        print("  Status    : ✗  DIRECTORY NOT FOUND")
    else:
        status = "✓  POPULATED" if store["count"] > 0 else "✗  EMPTY"
        print(f"  Status    : {status}")
        print(f"  Documents : {store['count']:,} JSON files")

    # ── Raw data ───────────────────────────────────────────────────────────
    _sep("RAW DATA (data/raw/)")
    raw = check_raw_dir()
    if not raw["exists"]:
        print("  Status    : ✗  DIRECTORY NOT FOUND")
    else:
        print(f"  PDF books : {raw['pdfs']}  (corpus inicial — libros médicos)")
        print(f"  JSONL     : {raw['jsonl']}  (salida del crawler web)")

    # ── Evaluation ─────────────────────────────────────────────────────────
    _sep("EVALUATION DATASET")
    ev = check_eval_dataset()
    status = "✓  AVAILABLE" if ev["ready"] else "✗  MISSING"
    print(f"  Status    : {status}")
    print(f"  Queries   : {'✓' if ev['queries'] else '✗'}  data/evaluation/eval_queries.json")
    print(f"  Qrels     : {'✓' if ev['qrels'] else '✗'}  data/evaluation/eval_qrels.json")

    _sep()

    # ── Overall readiness ──────────────────────────────────────────────────
    model_ok = lsi["ready"]
    data_ok = chroma["count"] > 0 and store["count"] > 0
    needs_ingest = raw.get("needs_ingest", False)

    if model_ok and data_ok:
        print("  ► SISTEMA LISTO — puede ejecutar consultas.")
    elif not model_ok and not data_ok and needs_ingest:
        print("  ► DATOS ELIMINADOS — corpus PDF sin procesar.")
        print()
        print("    Paso 1: extraer texto de los PDFs:")
        print("      python scripts/ingest_pdfs.py")
        print()
        print("    Paso 2: indexar corpus y entrenar LSI:")
        print("      python cli.py --stats")
    elif not model_ok and not data_ok:
        print("  ► DATOS ELIMINADOS — listo para demo desde cero.")
        print("    Ejecute:  python cli.py --stats")
    elif not model_ok:
        if needs_ingest:
            print("  ► PDFs sin procesar. Ejecute primero:")
            print("      python scripts/ingest_pdfs.py")
            print("    Luego:  python cli.py --stats")
        else:
            print("  ► Modelo LSI no entrenado. Ejecute:  python cli.py --stats")
    else:
        print("  ► Datos parciales. Considere ejecutar demo_reset.py")

    print()


if __name__ == "__main__":
    main()
