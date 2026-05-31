"""Content-based recommender module (optional §4.2.3).

Public API:
    :class:`ContentBasedRecommender` — LSI-similarity recommender with MMR
        diversification and an optional profile-aware source boost.
    :class:`RecommendedDocument`     — output container.
"""

from .service import ContentBasedRecommender, RecommendedDocument

__all__ = ["ContentBasedRecommender", "RecommendedDocument"]
