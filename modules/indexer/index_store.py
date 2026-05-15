"""Persistence and management for indexer artifacts.

Provides save/load/manage operations for the outputs of ``IndexerService`` so
that indexation state survives process restarts and can drive incremental
updates without re-processing every document from scratch.

Layout on disk (under a single ``storage_dir``):
    corpus.joblib      # full IndexedCorpus (documents, postings, vocabulary)
    spell_vocab.txt    # one processed token per line (restores the Trie)
    doc_ids.txt        # indexed doc_ids, one per line (fast membership lookup)
    manifest.json      # schema version, counts, created_at / updated_at

All writes are atomic: data is written to a ``.tmp`` sibling and then renamed
over the final file, so a crash mid-write never corrupts the existing snapshot.

Usage:
    store = IndexStore("data/indexer")

    # Persist after indexing
    store.save(corpus)
    store.save_spell_vocabulary(processor.spell_checker)

    # Restore on next run
    corpus = store.load()
    store.load_spell_vocabulary(processor.spell_checker)

    # Inspect / manage
    print(store.manifest())         # {"n_documents": ..., "updated_at": ...}
    print(store.indexed_doc_ids())  # {"d1", "d2", ...}
    store.clear()                   # wipe everything
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

from core.interfaces import IndexedCorpus
from modules.text_processor import TrieSpellChecker


logger = logging.getLogger(__name__)

_CORPUS_FILE = "corpus.joblib"
_SPELL_VOCAB_FILE = "spell_vocab.txt"
_DOC_IDS_FILE = "doc_ids.txt"
_MANIFEST_FILE = "manifest.json"
_SCHEMA_VERSION = "1.0"


class IndexStoreError(Exception):
    """Raised when an index artifact cannot be read or written."""

    pass


class IndexStore:
    """Persists and manages IndexerService outputs on the local filesystem.

    Separation of concerns:
        - IndexerService produces IndexedCorpus objects (pure computation).
        - IndexStore owns their lifecycle on disk (I/O + metadata).

    Every ``save(corpus)`` writes:
        1. ``corpus.joblib`` — the serialized IndexedCorpus.
        2. ``doc_ids.txt``  — the list of indexed doc_ids (lookup without load).
        3. ``manifest.json`` — refreshed counts and timestamp.

    ``save_spell_vocabulary()`` is a separate call so that the two artifacts
    can be written independently (e.g., spell vocabulary updated mid-session
    while the corpus snapshot is rewritten less frequently).

    Example:
        store = IndexStore("data/indexer")

        if store.exists():
            corpus = store.load()
            store.load_spell_vocabulary(processor.spell_checker)
            new_docs = [d for d in candidates if d.doc_id not in store.indexed_doc_ids()]
            corpus = indexer.update(corpus, new_docs)
        else:
            corpus = indexer.build(candidates)

        store.save(corpus)
        store.save_spell_vocabulary(processor.spell_checker)
    """

    def __init__(self, storage_dir: str | Path = "data/indexer") -> None:
        """Initialize the index store.

        Args:
            storage_dir: Directory where artifacts are written. Created if missing.

        Raises:
            OSError: If the directory cannot be created.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Corpus persistence
    # ------------------------------------------------------------------

    def save(self, corpus: IndexedCorpus) -> None:
        """Persist an IndexedCorpus snapshot atomically.

        Writes the corpus, the doc_id index file, and refreshes the manifest.

        Args:
            corpus: IndexedCorpus produced by IndexerService.

        Raises:
            IndexStoreError: If any artifact cannot be written.
        """
        self._atomic_joblib_write(self._corpus_path, corpus)

        doc_ids = [doc.doc_id for doc in corpus.documents]
        self._atomic_text_write(self._doc_ids_path, "\n".join(doc_ids))

        self._write_manifest(
            n_documents=len(corpus.documents),
            n_terms=len(corpus.vocabulary),
        )

        logger.info(
            "Saved IndexedCorpus: %d docs, %d terms → %s",
            len(corpus.documents), len(corpus.vocabulary), self.storage_dir,
        )

    def load(self) -> IndexedCorpus:
        """Load the last saved IndexedCorpus.

        Returns:
            Deserialized IndexedCorpus.

        Raises:
            IndexStoreError: If no corpus is saved or the file cannot be read.
        """
        if not self._corpus_path.exists():
            raise IndexStoreError(f"No corpus found at {self._corpus_path}")

        try:
            corpus: IndexedCorpus = joblib.load(self._corpus_path)
        except Exception as e:
            raise IndexStoreError(f"Cannot read corpus: {e}") from e

        logger.info(
            "Loaded IndexedCorpus: %d docs, %d terms from %s",
            len(corpus.documents), len(corpus.vocabulary), self.storage_dir,
        )
        return corpus

    def exists(self) -> bool:
        """Check whether a corpus snapshot is available on disk."""
        return self._corpus_path.exists()

    def clear(self) -> None:
        """Delete every artifact in the store (corpus, trie, doc_ids, manifest).

        Silently ignores files that do not exist. Useful when resetting the
        index for a clean re-crawl + re-indexation.
        """
        for path in (
            self._corpus_path,
            self._spell_path,
            self._doc_ids_path,
            self._manifest_path,
        ):
            if path.exists():
                path.unlink()
        logger.info("Cleared index store at %s", self.storage_dir)

    # ------------------------------------------------------------------
    # Spell vocabulary persistence
    # ------------------------------------------------------------------

    def save_spell_vocabulary(self, checker: TrieSpellChecker) -> None:
        """Persist the Trie vocabulary so queries can still be corrected later.

        The Trie itself is rebuilt on load via ``TrieSpellChecker.fit(words)``;
        only the flat list of words is serialized. Words are sorted so that
        the file diffs cleanly across saves.

        Args:
            checker: TrieSpellChecker whose vocabulary should be persisted.

        Raises:
            IndexStoreError: If the file cannot be written.
        """
        words = sorted(checker.words())
        self._atomic_text_write(self._spell_path, "\n".join(words))
        logger.info(
            "Saved spell vocabulary: %d words → %s", len(words), self._spell_path,
        )

    def load_spell_vocabulary(self, checker: TrieSpellChecker) -> int:
        """Restore the Trie vocabulary into an existing spell checker.

        This does NOT reset the checker first — words are inserted on top of
        whatever is already present. Call ``TrieSpellChecker.__init__`` again
        if a clean slate is required.

        Args:
            checker: TrieSpellChecker instance to populate.

        Returns:
            Number of words inserted. Zero if no saved vocabulary was found.

        Raises:
            IndexStoreError: If the vocabulary file exists but cannot be read.
        """
        if not self._spell_path.exists():
            logger.warning(
                "No spell vocabulary found at %s (queries will fall back to "
                "the live session vocabulary only)",
                self._spell_path,
            )
            return 0

        try:
            raw = self._spell_path.read_text(encoding="utf-8")
        except OSError as e:
            raise IndexStoreError(f"Cannot read spell vocabulary: {e}") from e

        words = [w for w in raw.splitlines() if w]
        checker.fit(words)
        logger.info(
            "Restored spell vocabulary: %d words from %s",
            len(words), self._spell_path,
        )
        return len(words)

    # ------------------------------------------------------------------
    # Management / inspection
    # ------------------------------------------------------------------

    def manifest(self) -> dict:
        """Return the manifest metadata as a dict.

        Returns:
            Dict with keys ``schema_version``, ``created_at``, ``updated_at``,
            ``n_documents``, ``n_terms``. Empty dict if no manifest is written
            or if the file is corrupted.
        """
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Corrupted manifest at %s: %s", self._manifest_path, e)
            return {}

    def indexed_doc_ids(self) -> set[str]:
        """Return the set of doc_ids currently in the saved index.

        Uses the sidecar ``doc_ids.txt`` when available to avoid loading the
        full corpus. Falls back to the corpus file if the sidecar is missing
        (e.g., corpus was written by an older version).

        Returns:
            Set of doc_ids. Empty if nothing is saved.
        """
        if self._doc_ids_path.exists():
            try:
                raw = self._doc_ids_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("Cannot read %s: %s", self._doc_ids_path, e)
                return set()
            return {line for line in raw.splitlines() if line}

        if self._corpus_path.exists():
            corpus = self.load()
            return {doc.doc_id for doc in corpus.documents}

        return set()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _corpus_path(self) -> Path:
        return self.storage_dir / _CORPUS_FILE

    @property
    def _spell_path(self) -> Path:
        return self.storage_dir / _SPELL_VOCAB_FILE

    @property
    def _doc_ids_path(self) -> Path:
        return self.storage_dir / _DOC_IDS_FILE

    @property
    def _manifest_path(self) -> Path:
        return self.storage_dir / _MANIFEST_FILE

    def _write_manifest(self, n_documents: int, n_terms: int) -> None:
        """Refresh the manifest, preserving ``created_at`` across updates."""
        existing = self.manifest()
        now = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "n_documents": n_documents,
            "n_terms": n_terms,
        }
        self._atomic_text_write(
            self._manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False),
        )

    @staticmethod
    def _atomic_joblib_write(path: Path, obj: object) -> None:
        """Dump ``obj`` to ``path`` atomically via a temp file + rename."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            joblib.dump(obj, tmp, compress=3)
            tmp.replace(path)
        except OSError as e:
            if tmp.exists():
                tmp.unlink()
            raise IndexStoreError(f"Cannot write {path.name}: {e}") from e

    @staticmethod
    def _atomic_text_write(path: Path, content: str) -> None:
        """Write ``content`` to ``path`` atomically via a temp file + rename."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        except OSError as e:
            if tmp.exists():
                tmp.unlink()
            raise IndexStoreError(f"Cannot write {path.name}: {e}") from e
