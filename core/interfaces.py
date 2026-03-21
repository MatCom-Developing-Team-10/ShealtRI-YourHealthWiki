"""Abstract interfaces used by mandatory SRI modules."""

from abc import ABC, abstractmethod

from core.models import Query, RetrievedDocument


class BaseRetriever(ABC):
    """Contract for retriever implementations."""

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10) -> list[RetrievedDocument]:
        """Return the top_k most relevant documents for a query.

        Args:
            query: Incoming user query.
            top_k: Number of documents to return.

        Returns:
            Ranked list of retrieved documents.
        """
        raise NotImplementedError
