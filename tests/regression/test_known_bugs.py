"""Regression tests for bugs documented in ``docs/informe-tecnico-tests-corte1.md``.

Originally each of these tests carried an ``@pytest.mark.xfail(strict=True)``
marker so that fixing the bug would force the marker to be retired in the
same commit. Both bugs have since been fixed, so the tests have been
promoted to standard regression tests — they MUST pass.

If any of these tests fails again, the corresponding fix has regressed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from modules.retriever.lsi_model import LSIModel


# ---------------------------------------------------------------------------
# Bug #3 (fixed) — LSIModel.fit() must handle a single-document corpus.
#
# Before the fix: ``effective_k = min(n_components, n_terms-1, n_docs-1)``
# became 0 for ``n_docs == 1`` and TruncatedSVD rejected the parameter.
# The fix pads the matrix with a zero row so SVD has two samples to work
# with, then discards the padding row from the returned vectors. The
# fitted ``_svd`` instance is still usable for ``project_query``.
# ---------------------------------------------------------------------------


def test_lsi_model_handles_single_document_corpus():
    """A 1×N TF-IDF matrix should yield a single ≥1-dim latent vector."""
    rng = np.random.default_rng(0)
    matrix = csr_matrix(rng.random((1, 5), dtype=np.float32))

    model = LSIModel(n_components=3)
    vectors = model.fit(matrix)

    assert len(vectors) == 1
    assert len(vectors[0]) >= 1
    # The fitted model must remain usable for query projection.
    assert model.is_fitted is True


def test_lsi_model_empty_corpus_raises():
    """A 0×N matrix is still an error — the fix only relaxes the n_docs==1 case."""
    import pytest

    empty = csr_matrix((0, 5), dtype=np.float32)
    model = LSIModel(n_components=3)
    with pytest.raises(ValueError, match="at least one document"):
        model.fit(empty)


def test_lsi_model_single_doc_query_projection_works():
    """After fitting on 1 doc, project_query must return a vector of the same dim."""
    rng = np.random.default_rng(1)
    doc_matrix = csr_matrix(rng.random((1, 5), dtype=np.float32))
    query_matrix = csr_matrix(rng.random((1, 5), dtype=np.float32))

    model = LSIModel(n_components=3)
    doc_vectors = model.fit(doc_matrix)
    query_vector = model.project_query(query_matrix)

    assert len(query_vector) == len(doc_vectors[0])


# ---------------------------------------------------------------------------
# Bug #4 (fixed) — modules.indexer must not transitively import spaCy.
#
# Before the fix:
#   modules.indexer.__init__ re-exported IndexStore, which imported
#   TrieSpellChecker via ``from modules.text_processor import TrieSpellChecker``,
#   which loaded ``modules/text_processor/__init__.py``, which imported
#   ``service.py``, which imported spaCy.
#
# The fix has two parts:
#   1. ``index_store.py`` now imports TrieSpellChecker from the submodule
#      directly: ``from modules.text_processor.spell_checker import …``
#   2. ``modules/indexer/__init__.py`` exposes ``IndexerService`` and
#      ``IndexStore`` via PEP-562 ``__getattr__`` so they load lazily only
#      when explicitly requested by name.
# ---------------------------------------------------------------------------


def test_indexer_subpackage_does_not_pull_spacy(tmp_path):
    """Importing modules.indexer must not import spaCy.

    Runs inside a subprocess with a meta-path blocker that raises
    ImportError on any ``import spacy``/``import spacy.*`` attempt, so a
    side-effect import would surface as a non-zero exit code.
    """
    project_root = Path(__file__).resolve().parents[2]
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'spacy' or name.startswith('spacy.'):\n"
        "            raise ImportError('spaCy was imported as a side-effect of `import modules.indexer`')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "import modules.indexer\n"
        "from modules.indexer.document_store import FileSystemDocumentStore\n",
        encoding="utf-8",
    )

    # ``cwd`` alone does not put the project root on ``sys.path`` for a
    # subprocess invoked with a script path; PYTHONPATH does.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Importing modules.indexer pulled spaCy.\nstderr:\n{result.stderr}"
    )


def test_indexer_lazy_attrs_still_resolvable():
    """The lazy ``__getattr__`` plumbing must keep the public API working."""
    import modules.indexer as pkg

    # Eager attribute — present from import.
    assert pkg.FileSystemDocumentStore is not None
    # Lazy attributes — resolved on demand.
    assert pkg.IndexerService is not None
    assert pkg.IndexStore is not None
    assert pkg.IndexerConfig is not None

    # And listed via ``dir`` so introspection tools find them.
    listed = dir(pkg)
    for name in ("IndexerService", "IndexStore", "IndexerConfig", "FileSystemDocumentStore"):
        assert name in listed, f"{name!r} missing from dir(modules.indexer)"


def test_indexer_lazy_unknown_attr_raises_attribute_error():
    """``__getattr__`` must still raise AttributeError for unknown names."""
    import pytest

    import modules.indexer as pkg

    with pytest.raises(AttributeError):
        pkg.does_not_exist  # noqa: B018
