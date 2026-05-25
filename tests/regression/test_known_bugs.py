"""Regression tests for documented bugs that are NOT yet fixed.

Each test is marked ``@pytest.mark.xfail(strict=True)`` with the bug ID
from the technical report. When the bug is fixed:

    - The test will start passing.
    - ``strict=True`` turns the unexpected pass into a test failure.
    - The committer is then forced to either:
        a) remove the xfail marker (preferred — the bug is fixed); or
        b) explain why the bug came back.

This way the test suite tracks the bug backlog automatically.
"""

from __future__ import annotations

import pytest
import numpy as np
from scipy.sparse import csr_matrix

from modules.retriever.lsi_model import LSIModel


# ---------------------------------------------------------------------------
# Bug #3 from informe-tecnico — LSIModel.fit() crashes with n_docs == 1
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bug #3 in docs/informe-tecnico-tests-corte1.md: "
        "LSIModel.fit() computes effective_k = min(n_components, n_terms-1, n_docs-1). "
        "With n_docs == 1 this becomes 0 and TruncatedSVD rejects the value. "
        "Expected fix: effective_k = max(1, min(...)). When that lands, this test "
        "starts passing — REMOVE the xfail marker in the same commit."
    ),
)
def test_lsi_model_handles_single_document_corpus():
    """When fixed, fitting on a single-doc corpus should produce a 1-dim embedding."""
    rng = np.random.default_rng(0)
    matrix = csr_matrix(rng.random((1, 5), dtype=np.float32))
    model = LSIModel(n_components=3)
    vectors = model.fit(matrix)  # currently raises InvalidParameterError
    assert len(vectors) == 1
    # After fix, n_components is clamped to >= 1
    assert len(vectors[0]) >= 1


# ---------------------------------------------------------------------------
# Bug #4 — modules.indexer import chain pulls in spaCy
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bug #4 in docs/informe-tecnico-tests-corte1.md: "
        "Importing modules.indexer transitively loads modules.text_processor.__init__, "
        "which imports spaCy. FileSystemDocumentStore should be importable without spaCy. "
        "Expected fix: modules/indexer/index_store.py imports TrieSpellChecker directly "
        "from modules.text_processor.spell_checker (bypassing the package __init__)."
    ),
)
def test_indexer_subpackage_does_not_pull_spacy(tmp_path):
    """When fixed, importing modules.indexer must not import spaCy.

    Run inside a subprocess so we do NOT pollute the parent process's
    ``sys.modules`` (which would corrupt the session-scoped TextProcessor
    fixture used by other tests).
    """
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        # Block spaCy at the importer level before anything project-related loads
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

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Importing modules.indexer pulled spaCy.\n"
        f"stderr:\n{result.stderr}"
    )
