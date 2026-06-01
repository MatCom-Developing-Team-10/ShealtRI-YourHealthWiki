"""Tests for the plugin orchestrator (microkernel hook engine)."""

from __future__ import annotations

import pytest

from core.interfaces import Plugin
from core.models import PipelineContext, Query
from core.plugin_pipeline import VALID_HOOKS, PluginPipeline


class _TracePlugin(Plugin):
    """Records its tag in context.metadata['trace'] when executed."""

    def __init__(self, hook: str, tag: str) -> None:
        self._hook = hook
        self._tag = tag

    def hook_name(self) -> str:
        return self._hook

    def execute(self, context: PipelineContext) -> PipelineContext:
        context.metadata.setdefault("trace", []).append(self._tag)
        return context


class _FailingPlugin(Plugin):
    def hook_name(self) -> str:
        return "pre_retrieval"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raise RuntimeError("boom")


def _context() -> PipelineContext:
    return PipelineContext(query=Query(text="diabetes"))


class TestRegistration:
    def test_groups_by_declared_hook(self):
        pipe = PluginPipeline([_TracePlugin("pre_retrieval", "a")])
        assert pipe.has_plugins("pre_retrieval")
        assert not pipe.has_plugins("post_ranking")

    def test_invalid_hook_raises(self):
        class BadPlugin(_TracePlugin):
            def hook_name(self) -> str:
                return "not_a_hook"

        with pytest.raises(ValueError, match="unknown hook"):
            PluginPipeline([BadPlugin("x", "y")])

    def test_all_valid_hooks_accepted(self):
        plugins = [_TracePlugin(h, h) for h in VALID_HOOKS]
        pipe = PluginPipeline(plugins)
        for hook in VALID_HOOKS:
            assert pipe.has_plugins(hook)


class TestRunHook:
    def test_executes_plugins_in_registration_order(self):
        pipe = PluginPipeline(
            [_TracePlugin("pre_retrieval", "first"),
             _TracePlugin("pre_retrieval", "second")]
        )
        ctx = pipe.run_hook("pre_retrieval", _context())
        assert ctx.metadata["trace"] == ["first", "second"]

    def test_only_runs_plugins_for_that_hook(self):
        pipe = PluginPipeline(
            [_TracePlugin("pre_retrieval", "pre"),
             _TracePlugin("post_ranking", "post")]
        )
        ctx = pipe.run_hook("pre_retrieval", _context())
        assert ctx.metadata["trace"] == ["pre"]

    def test_unknown_hook_is_noop(self):
        pipe = PluginPipeline([_TracePlugin("pre_retrieval", "a")])
        ctx = _context()
        result = pipe.run_hook("nonexistent_hook", ctx)
        assert "trace" not in result.metadata

    def test_empty_pipeline_returns_context_unchanged(self):
        pipe = PluginPipeline()
        ctx = _context()
        assert pipe.run_hook("pre_retrieval", ctx) is ctx

    def test_failing_plugin_is_skipped_not_fatal(self):
        # A failing plugin must not stop later plugins at the same hook.
        pipe = PluginPipeline(
            [_FailingPlugin(), _TracePlugin("pre_retrieval", "survivor")]
        )
        ctx = pipe.run_hook("pre_retrieval", _context())
        assert ctx.metadata["trace"] == ["survivor"]


class TestExpansionPluginIntegration:
    def test_expansion_plugin_runs_through_pipeline(self):
        # The real expansion plugin must be registerable and runnable via the
        # orchestrator without error, recording its breadcrumb in metadata.
        from collections import Counter

        from core.interfaces import IndexedCorpus
        from core.models import Document
        from plugins.expansion.service import QueryExpansionPlugin

        tokens = ["diabetes", "glucosa"]
        inv = {t: [(0, tf)] for t, tf in Counter(tokens).items()}
        query_corpus = IndexedCorpus(
            documents=[Document("__query__", " ".join(tokens), "")],
            processed_texts=[" ".join(tokens)],
            inverted_index=inv,
            vocabulary=sorted(inv.keys()),
        )
        query = Query(text="diabetes glucosa", indexed_corpus=query_corpus)

        pipe = PluginPipeline(
            [QueryExpansionPlugin(target_vocabulary=["diabetes", "glucosa", "azucar"])]
        )
        ctx = pipe.run_hook("pre_retrieval", PipelineContext(query=query))

        # Plugin executed: it left its expansion breadcrumb.
        assert "expansion" in ctx.metadata
        assert ctx.query.indexed_corpus is not None
