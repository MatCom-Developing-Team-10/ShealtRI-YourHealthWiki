"""Retriever context that applies the Strategy Pattern."""

from core.interfaces import BaseRetriever
from core.models import Query, RetrievedDocument


class RetrievalContext:
    """Class that uses a BaseRetriever strategy to perform searches."""

    def __init__(self, strategy: BaseRetriever) -> None:
        """Initialize with a specific retrieval strategy.

        Args:
            strategy: Concrete implementation of BaseRetriever.
        """
        self._strategy = strategy

    @property
    def strategy(self) -> BaseRetriever:
        """Get the current strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseRetriever) -> None:
        """Switch the strategy at runtime."""
        self._strategy = strategy

    def execute_search(self, query: Query, top_k: int = 10) -> list[RetrievedDocument]:
        """Execute the current retrieval strategy.

        Args:
            query: User's query.
            top_k: Number of results to return.

        Returns:
            List of relevant documents.
        """
        return self._strategy.retrieve(query, top_k=top_k)
