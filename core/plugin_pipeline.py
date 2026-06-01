"""Plugin orchestrator — the microkernel that runs optional plugins at hooks.

The core never imports a concrete plugin. Instead, plugins implementing
:class:`core.interfaces.Plugin` are registered here and executed at their
declared hook points. If no plugin is registered for a hook, that hook is a
no-op, so the pipeline behaves exactly as if plugins did not exist.

Hook order during a query:

    pre_retrieval  → (retriever runs) → post_retrieval → (ranker runs) → post_ranking

Usage:
    pipeline = PluginPipeline([QueryExpansionPlugin(target_vocabulary=vocab)])
    context = PipelineContext(query=query)
    context = pipeline.run_hook("pre_retrieval", context)
    # ... retriever uses context.query (possibly expanded) ...
"""

from __future__ import annotations

import logging

from core.interfaces import Plugin
from core.models import PipelineContext

logger = logging.getLogger(__name__)

# The hook points recognised by the pipeline, in execution order.
VALID_HOOKS = ("pre_retrieval", "post_retrieval", "post_ranking")


class PluginPipeline:
    """Registers plugins by hook and runs them on a shared PipelineContext.

    Args:
        plugins: Plugins to register. Each is grouped by its ``hook_name()``.
            Registration order within a hook is preserved as execution order.

    Raises:
        ValueError: If a plugin declares a hook outside :data:`VALID_HOOKS`.
    """

    def __init__(self, plugins: list[Plugin] | None = None) -> None:
        self._by_hook: dict[str, list[Plugin]] = {hook: [] for hook in VALID_HOOKS}
        for plugin in plugins or []:
            self.register(plugin)

    def register(self, plugin: Plugin) -> None:
        """Register a single plugin at its declared hook.

        Args:
            plugin: The plugin instance to add.

        Raises:
            ValueError: If the plugin's hook is not recognised.
        """
        hook = plugin.hook_name()
        if hook not in self._by_hook:
            raise ValueError(
                f"Plugin {type(plugin).__name__} declares unknown hook {hook!r}; "
                f"expected one of {VALID_HOOKS}"
            )
        self._by_hook[hook].append(plugin)
        logger.debug("Registered %s at hook %r", type(plugin).__name__, hook)

    def has_plugins(self, hook: str) -> bool:
        """Whether any plugin is registered at *hook*."""
        return bool(self._by_hook.get(hook))

    def run_hook(self, hook: str, context: PipelineContext) -> PipelineContext:
        """Run every plugin registered at *hook*, in registration order.

        Each plugin receives the context returned by the previous one. A plugin
        that raises is logged and skipped so one faulty plugin cannot break the
        whole query (plugins are optional by contract).

        Args:
            hook: The hook point to run. Unknown hooks are a no-op.
            context: The pipeline context to transform.

        Returns:
            The context after all plugins at this hook have run.
        """
        for plugin in self._by_hook.get(hook, []):
            try:
                context = plugin.execute(context)
            except Exception as exc:  # plugins are optional — never fatal
                logger.warning(
                    "Plugin %s failed at hook %r: %s; skipping",
                    type(plugin).__name__, hook, exc,
                )
        return context
