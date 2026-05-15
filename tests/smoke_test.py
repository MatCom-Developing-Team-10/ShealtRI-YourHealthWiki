"""Smoke tests — run with plain Python, no external dependencies required.

    python tests/smoke_test.py

These tests stub out spaCy, ChromaDB, and scikit-learn using only
unittest.mock (Python stdlib), then exercise the pure-Python logic:
  - core.models   (Document, Query, RetrievedDocument)
  - core.interfaces (IndexedCorpus invariant)
  - core.pipeline  (RetrievalContext strategy wrapper)
  - modules.indexer.service (build, stats, update, remove — pure Python logic)

What this CANNOT test (requires real deps):
  - TextProcessor NLP (lemmatisation, spell correction — needs spaCy)
  - LSIRetriever.fit / retrieve (needs scikit-learn)
  - ChromaRepository vector search (needs chromadb)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Add project root to sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# 2. Stub heavy external modules BEFORE any project import loads them
# ---------------------------------------------------------------------------
import types

def _pkg(name: str, **attrs) -> types.ModuleType:
    """Create a real ModuleType so submodule attribute access works."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod

# --- spaCy: needs spacy.language.Language as a real class ---
_Language = type("Language", (), {})
_spacy_language = _pkg("spacy.language", Language=_Language)
_spacy = _pkg("spacy", language=_spacy_language, Language=_Language)
_spacy.load = MagicMock(return_value=MagicMock(return_value=MagicMock(__iter__=lambda s: iter([]))))
_spacy_lang = _pkg("spacy.lang")
_spacy_lang_es = _pkg("spacy.lang.es")

sys.modules["spacy"] = _spacy
sys.modules["spacy.language"] = _spacy_language
sys.modules["spacy.lang"] = _spacy_lang
sys.modules["spacy.lang.es"] = _spacy_lang_es

# --- NLTK: needs nltk.corpus with a stopwords attribute ---
_nltk_corpus = _pkg("nltk.corpus")
_nltk_corpus.stopwords = MagicMock()
_nltk_corpus.stopwords.words = MagicMock(return_value=[])
_nltk = _pkg("nltk", corpus=_nltk_corpus)
_nltk.download = MagicMock()
sys.modules["nltk"] = _nltk
sys.modules["nltk.corpus"] = _nltk_corpus

# --- Simple MagicMock stubs for everything else ---
for _mod in [
    "joblib",
    "chromadb",
    "sklearn", "sklearn.decomposition",
    "sklearn.feature_extraction", "sklearn.feature_extraction.text",
    "langchain", "langchain.schema",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.document_loaders.base",
    "sentence_transformers",
]:
    sys.modules.setdefault(_mod, MagicMock())

# ---------------------------------------------------------------------------
# 3. Minimal TextProcessor stub (no spaCy, pure Python)
# ---------------------------------------------------------------------------

class _StubTextProcessor:
    """Tokenises by lowercasing and splitting — no NLP needed."""

    def process(self, text: str, is_query: bool = False) -> str:
        tokens = []
        for raw in text.lower().split():
            tok = raw.strip(".,;:!?\"'()[]{}")
            if len(tok) >= 3:
                tokens.append(tok)
        return " ".join(tokens)


# ---------------------------------------------------------------------------
# 4. Now import project modules (stubs are already in sys.modules)
# ---------------------------------------------------------------------------

from core.interfaces import IndexedCorpus          # noqa: E402
from core.models import Document, Query, RetrievedDocument  # noqa: E402
from core.pipeline import RetrievalContext         # noqa: E402
from modules.indexer.service import IndexerService, IndexerConfig  # noqa: E402


# ---------------------------------------------------------------------------
# 5. Test cases
# ---------------------------------------------------------------------------

class TestDataModels(unittest.TestCase):
    """core.models — no external deps at all."""

    def test_document_creation(self):
        doc = Document(doc_id="d1", text="diabetes tipo 2", url="http://x.com")
        self.assertEqual(doc.doc_id, "d1")
        self.assertEqual(doc.text, "diabetes tipo 2")
        self.assertEqual(doc.metadata, {})

    def test_query_defaults(self):
        q = Query(text="hipertensión")
        self.assertIsNone(q.indexed_corpus)
        self.assertEqual(q.metadata, {})

    def test_retrieved_document(self):
        doc = Document(doc_id="d1", text="texto", url="")
        rd = RetrievedDocument(document=doc, score=0.75)
        self.assertEqual(rd.score, 0.75)
        self.assertIs(rd.document, doc)


class TestIndexedCorpusInvariant(unittest.TestCase):
    """core.interfaces.IndexedCorpus enforces alignment invariant."""

    def test_valid_corpus(self):
        docs = [Document(doc_id=f"d{i}", text="x", url="") for i in range(3)]
        corpus = IndexedCorpus(
            documents=docs,
            processed_texts=["a", "b", "c"],
            inverted_index={"a": [(0, 1)]},
            vocabulary=["a"],
        )
        self.assertEqual(len(corpus), 3)

    def test_invariant_raises_on_mismatch(self):
        docs = [Document(doc_id="d1", text="x", url="")]
        with self.assertRaises(ValueError):
            IndexedCorpus(
                documents=docs,
                processed_texts=["a", "b"],  # length mismatch
                inverted_index={},
                vocabulary=[],
            )

    def test_doc_ids_property(self):
        docs = [Document(doc_id=f"id_{i}", text="x", url="") for i in range(2)]
        corpus = IndexedCorpus(
            documents=docs,
            processed_texts=["t", "t"],
            inverted_index={},
            vocabulary=[],
        )
        self.assertEqual(corpus.doc_ids, ["id_0", "id_1"])

    def test_empty_corpus_is_valid(self):
        corpus = IndexedCorpus(
            documents=[], processed_texts=[], inverted_index={}, vocabulary=[]
        )
        self.assertEqual(len(corpus), 0)


class TestIndexerService(unittest.TestCase):
    """IndexerService — pure Python logic, no spaCy/sklearn needed."""

    def setUp(self):
        self.processor = _StubTextProcessor()
        self.indexer = IndexerService(text_processor=self.processor)

    def _make_doc(self, doc_id: str, text: str) -> Document:
        return Document(doc_id=doc_id, text=text, url=f"http://example.com/{doc_id}")

    def test_build_produces_corpus(self):
        docs = [
            self._make_doc("d1", "hipertensión arterial presión sangre"),
            self._make_doc("d2", "diabetes glucosa insulina páncreas"),
            self._make_doc("d3", "asma bronquial respiratorio pulmones"),
        ]
        corpus = self.indexer.build(docs)
        self.assertEqual(len(corpus.documents), 3)
        self.assertEqual(len(corpus.processed_texts), 3)
        self.assertGreater(len(corpus.vocabulary), 0)

    def test_build_empty_returns_valid_corpus(self):
        corpus = self.indexer.build([])
        self.assertEqual(len(corpus.documents), 0)
        self.assertEqual(corpus.vocabulary, [])

    def test_corpus_length_invariant_after_build(self):
        docs = [self._make_doc(f"d{i}", f"texto médico número {i}") for i in range(5)]
        corpus = self.indexer.build(docs)
        self.assertEqual(len(corpus.documents), len(corpus.processed_texts))

    def test_stats_returns_expected_keys(self):
        docs = [self._make_doc(f"d{i}", f"enfermedad síntoma tratamiento {i}") for i in range(4)]
        corpus = self.indexer.build(docs)
        stats = IndexerService.stats(corpus)

        self.assertIn("n_documents", stats)
        self.assertIn("n_terms", stats)
        self.assertIn("total_tokens", stats)
        self.assertIn("avg_tokens_per_doc", stats)
        self.assertIn("avg_postings_per_term", stats)

    def test_stats_values_are_sane(self):
        docs = [self._make_doc(f"d{i}", f"diabetes glucosa insulina síntoma tratamiento {i}") for i in range(5)]
        corpus = self.indexer.build(docs)
        stats = IndexerService.stats(corpus)

        self.assertEqual(stats["n_documents"], 5)
        self.assertGreater(stats["n_terms"], 0)
        self.assertGreater(stats["avg_tokens_per_doc"], 0)

    def test_stats_on_empty_corpus(self):
        corpus = self.indexer.build([])
        stats = IndexerService.stats(corpus)
        self.assertEqual(stats["n_documents"], 0)
        self.assertEqual(stats["avg_tokens_per_doc"], 0.0)

    def test_remove_drops_document(self):
        docs = [self._make_doc(f"d{i}", f"texto específico para documento número {i}") for i in range(3)]
        corpus = self.indexer.build(docs)
        reduced = self.indexer.remove(corpus, ["d1"])
        ids = [d.doc_id for d in reduced.documents]
        self.assertNotIn("d1", ids)
        self.assertEqual(len(reduced.documents), 2)

    def test_remove_unknown_id_is_noop(self):
        docs = [self._make_doc(f"d{i}", f"texto para documento {i}") for i in range(2)]
        corpus = self.indexer.build(docs)
        same = self.indexer.remove(corpus, ["nonexistent"])
        self.assertEqual(len(same.documents), 2)

    def test_update_adds_new_document(self):
        docs = [self._make_doc(f"d{i}", f"texto inicial documento {i}") for i in range(2)]
        corpus = self.indexer.build(docs)
        new_doc = self._make_doc("d_new", "nuevo documento médico cardiovascular")
        updated = self.indexer.update(corpus, [new_doc])
        ids = [d.doc_id for d in updated.documents]
        self.assertIn("d_new", ids)
        self.assertEqual(len(updated.documents), 3)

    def test_update_skips_duplicate(self):
        docs = [self._make_doc(f"d{i}", f"texto documento {i}") for i in range(2)]
        corpus = self.indexer.build(docs)
        same_doc = self._make_doc("d0", "contenido diferente pero mismo id")
        updated = self.indexer.update(corpus, [same_doc])
        self.assertEqual(len(updated.documents), 2)

    def test_build_query_returns_single_doc_corpus(self):
        # Build first to populate spell checker vocab
        docs = [self._make_doc(f"d{i}", f"diabetes glucosa insulina {i}") for i in range(3)]
        self.indexer.build(docs)
        qc = self.indexer.build_query("glucosa insulina")
        self.assertEqual(len(qc.documents), 1)
        self.assertEqual(len(qc.processed_texts), 1)

    def test_build_query_empty_string(self):
        qc = self.indexer.build_query("")
        self.assertEqual(len(qc.documents), 1)


class TestRetrievalContext(unittest.TestCase):
    """core.pipeline.RetrievalContext — strategy pattern, no external deps."""

    def _make_mock_retriever(self, results):
        mock = MagicMock()
        mock.retrieve.return_value = results
        return mock

    def test_execute_search_delegates_to_strategy(self):
        doc = Document(doc_id="d1", text="diabetes", url="")
        expected = [RetrievedDocument(document=doc, score=0.9)]
        retriever = self._make_mock_retriever(expected)

        ctx = RetrievalContext(strategy=retriever)
        query = Query(text="diabetes")
        results = ctx.execute_search(query, top_k=5)

        retriever.retrieve.assert_called_once_with(query, top_k=5)
        self.assertEqual(results, expected)

    def test_strategy_can_be_switched_at_runtime(self):
        r1 = self._make_mock_retriever([])
        r2 = self._make_mock_retriever([])
        ctx = RetrievalContext(strategy=r1)
        self.assertIs(ctx.strategy, r1)

        ctx.strategy = r2
        self.assertIs(ctx.strategy, r2)


# ---------------------------------------------------------------------------
# 6. Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [
        TestDataModels,
        TestIndexedCorpusInvariant,
        TestIndexerService,
        TestRetrievalContext,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
