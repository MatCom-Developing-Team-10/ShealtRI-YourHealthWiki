"""Optional plugin packages that extend the SRI pipeline.

Plugins implement :class:`core.interfaces.Plugin` and are wired into the
pipeline at well-known hook points (``pre_retrieval``, ``post_retrieval``,
``post_ranking``). The core never imports a plugin directly; if a plugin
is not registered, the pipeline behaves as if it did not exist.
"""
