from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import List, Tuple


class EnterpriseRetriever:
    """
    Wrapper class for semantic retrieval from the Chroma vector store.
    """

    def __init__(self, vectorstore: Chroma, k: int = 5):
        # Store the underlying vector database and default top-k retrieval count.
        self.vectorstore = vectorstore
        self.k = k

    def as_retriever(self):
        # Expose a LangChain-compatible retriever when needed by other components.
        return self.vectorstore.as_retriever(search_kwargs={"k": self.k})

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Standard similarity search used for retrieval-only flows.
        return self.vectorstore.similarity_search(query, k=self.k)

    def get_relevant_documents_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        # Returns both documents and similarity scores for quality assessment.
        return self.vectorstore.similarity_search_with_score(query, k=self.k)

    def has_sufficient_context(self, query: str, max_score_threshold: float = 1.2) -> bool:
        """
        Lower score generally means closer match in Chroma distance-based search.
        This threshold may need tuning based on embedding model behavior.
        """
        results = self.get_relevant_documents_with_scores(query)
        if not results:
            return False

        # Use the best match to decide whether the retrieved context is likely usable.
        best_score = min(score for _, score in results)
        return best_score <= max_score_threshold
