from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import List


class EnterpriseRetriever:
    """
    Wrapper class for semantic retrieval from the Chroma vector store.

    Why this class exists:
    - It provides a clean interface between the vector database and
      the agentic RAG pipeline.
    - It keeps retrieval logic separate from ingestion, embedding,
      and generation logic.
    - It supports the capstone task of retrieving the most relevant
      document content based on a user query. [file:24]
    """

    def __init__(self, vectorstore: Chroma):
        """
        Initialize the retriever with an existing Chroma vector store.

        Args:
            vectorstore: A Chroma database instance containing embedded
                         document chunks.
        """
        self.vectorstore = vectorstore

    def as_retriever(self):
        """
        Return the vector store as a LangChain-compatible retriever object.

        Why this is useful:
        - Some LangChain pipelines expect a retriever interface instead
          of direct vector store calls.
        - This makes the class more reusable if the project is later extended
          with built-in LangChain retrieval chains.

        Returns:
            A retriever object configured to return the top 5 matches.
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve the most relevant document chunks for a given query.

        How it works:
        - The query is converted into an embedding internally.
        - Chroma compares that query embedding with stored document embeddings.
        - The most semantically similar chunks are returned.

        Args:
            query: User's natural language question.

        Returns:
            A list of the top 5 most relevant document chunks.
        """
        return self.vectorstore.similarity_search(query, k=5)
