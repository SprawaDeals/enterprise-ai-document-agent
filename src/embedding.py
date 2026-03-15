from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import settings
from src.ingestion import load_document, chunk_text
from typing import List
import os


# Initialize the embedding model used to convert document chunks into vectors.
# These vectors are stored in ChromaDB and later used for semantic similarity search.
# The model name is read from the central settings file for easy maintenance.
embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_vector_store(docs_dir: str) -> Chroma:
    """
    Build or load the Chroma vector store for enterprise documents.

    Main workflow:
    1. Validate the input folder.
    2. Load an existing vector store if already created.
    3. If no vector store exists, read documents from the folder.
    4. Convert document text into chunks.
    5. Generate embeddings for each chunk.
    6. Store embeddings in ChromaDB for semantic retrieval.

    This function implements the capstone requirements for:
    - document ingestion,
    - preparing data for semantic search,
    - building a vector-based knowledge store. [file:24]

    Args:
        docs_dir: Path to the directory containing input documents.

    Returns:
        A Chroma vector store object that can be used for retrieval.
    """
    # Validate that the document directory exists before processing.
    # This avoids runtime failure later when trying to read files.
    if not os.path.exists(docs_dir):
        raise ValueError(f"Docs directory '{docs_dir}' not found. Add files to ./data/")

    # Try loading an already existing vector store first.
    # This makes the function reusable and avoids regenerating embeddings
    # every time the application starts.
    try:
        vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings
        )
        print("Loaded existing vector store.")
        return vectorstore

    except Exception:
        # If loading fails, create a new vector store from documents.
        # This usually happens the first time the project is run.
        print("Creating new vector store...")
        vectorstore = None

    # Loop through all supported files in the document folder.
    # Supported formats are based on the capstone requirement:
    # PDF, TXT, CSV, and Excel.[file:24]
    for file in os.listdir(docs_dir):
        if file.endswith(('.pdf', '.txt', '.csv', '.xlsx')):
            file_path = os.path.join(docs_dir, file)
            print(f"Processing {file}...")

            # Read the raw text or tabular content from the file.
            # The ingestion logic is handled in src.ingestion.
            text = load_document(file_path)

            # Split the full document content into smaller overlapping chunks.
            # Chunking is necessary because embedding models and RAG pipelines
            # work better with smaller semantically coherent pieces of text.
            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)

            # Convert each chunk into a LangChain Document object.
            # Metadata stores the original source filename, which is useful
            # for traceability and citation in generated answers.
            documents = [
                Document(page_content=chunk, metadata={"source": file})
                for chunk in chunks
            ]

            # If this is the first batch of documents, create a new Chroma store.
            # Chroma.from_documents both embeds the text and stores it.
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents,
                    embeddings,
                    persist_directory=settings.chroma_persist_dir
                )
            else:
                # For additional files, add their document chunks to the
                # existing vector store. Chroma auto-saves when using
                # persist_directory in recent versions.
                vectorstore.add_documents(documents)

    # Final confirmation that the vector store is ready for query-time retrieval.
    print(f"✅ Vector store ready at {settings.chroma_persist_dir}")
    return vectorstore
