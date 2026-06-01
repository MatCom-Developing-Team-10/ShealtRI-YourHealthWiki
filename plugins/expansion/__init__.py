"""Query expansion plugin for the SRI pipeline.

Public API:
    :class:`QueryExpander`         — pure expansion logic (medical thesaurus).
    :class:`QueryExpansionPlugin`  — wraps QueryExpander into a Plugin
                                     registered at the ``pre_retrieval`` hook.
    :class:`ExpansionConfig`       — knobs for the expansion strategy.
    :data:`MEDICAL_THESAURUS`      — Spanish medical term/synonym map.
"""

from .service import (
    ExpansionConfig,
    QueryExpander,
    QueryExpansionPlugin,
)
from .thesaurus import MEDICAL_THESAURUS, expand_term

__all__ = [
    "ExpansionConfig",
    "QueryExpander",
    "QueryExpansionPlugin",
    "MEDICAL_THESAURUS",
    "expand_term",
]
