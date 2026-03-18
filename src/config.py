from pydantic_settings import BaseSettings  # BaseSettings is used for environment-based configuration in Pydantic v2
from typing import Optional
import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment.
# This allows sensitive values like API keys to be stored outside the source code.
load_dotenv()


class Settings(BaseSettings):
    """
    Central configuration class for the enterprise AI agent system.

    Why this is useful:
    - Keeps all configurable values in one place.
    - Makes the project easier to maintain and deploy.
    - Supports environment-based configuration, which is good practice
      for local development, testing, and deployment.

    This aligns with Task 1 of the capstone:
    setting up the project foundation and environment configuration. [file:24]
    """

    # API key used by the OpenAI LLM client.
    # Loaded from the environment for security instead of hardcoding it.
    openai_api_key: str = os.getenv("OPENAI_API_KEY")

    # Embedding model used to convert document chunks into vectors.
    # This is required for semantic search in the vector database.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Main language model used for planning, reasoning, and validation.
    # This powers the Generative AI response layer in the RAG pipeline.
    llm_model: str = "gpt-4o-mini"

    # Folder path where the Chroma vector database will be stored locally.
    # This allows the system to persist embeddings between application runs.
    chroma_persist_dir: str = "./chroma_db"

    # Maximum number of characters in each chunk.
    # Chunking is necessary because long documents must be split into smaller
    # sections before embedding and retrieval.
    chunk_size: int = 1000

    # Number of overlapping characters between chunks.
    # Overlap helps preserve continuity of meaning across chunk boundaries.
    chunk_overlap: int = 200

    # Number of most relevant chunks to retrieve for each query.
    # This controls how much context is passed into the RAG pipeline.
    top_k: int = 5

    # Maximum token budget for generated output.
    # This can help control response length and reduce unnecessary verbosity.
    max_new_tokens: int = 500

    max_agent_iterations: int = 3
    retrieval_k: int = 5
    retrieval_score_threshold: float = 1.2

    class Config:
        # Tells Pydantic to read configuration values from the .env file.
        env_file = ".env"


# Create a single settings object that can be imported across the application.
# This avoids repeating configuration logic in multiple files.
settings = Settings()
