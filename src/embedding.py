from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import settings
from src.ingestion import load_document, chunk_text
import os
import time


def build_vector_store(docs_dir: str, persist_dir: str) -> Chroma:
    """
    Build a Chroma vector store from all supported documents in the given folder.
    """
    # Debug logs help track where indexing is slow or failing.
    print(f"[1] Starting build_vector_store")
    print(f"[2] docs_dir = {docs_dir}")
    print(f"[3] persist_dir = {persist_dir}")

    if not os.path.exists(docs_dir):
        raise ValueError(f"Docs directory '{docs_dir}' not found.")

    # Load the embedding model once and reuse it for all document chunks.
    print("[4] Loading embedding model...")
    start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    print(f"[5] Embedding model loaded in {time.time() - start:.2f}s")

    all_documents = []

    # Restrict ingestion to supported enterprise document formats.
    files = [f for f in os.listdir(docs_dir) if f.endswith((".pdf", ".txt", ".csv", ".xlsx", ".xls"))]
    print(f"[6] Found {len(files)} supported files")

    for file in files:
        file_path = os.path.join(docs_dir, file)
        print(f"[7] Processing file: {file_path}")

        # Extract raw text from the source document.
        text = load_document(file_path)
        print(f"[8] Extracted text length for {file}: {len(text)}")

        # Split long text into retrieval-friendly chunks for embedding.
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        print(f"[9] Created {len(chunks)} chunks for {file}")

        # Store the source filename in metadata so downstream answers can be grounded.
        documents = [
            Document(page_content=chunk, metadata={"source": file})
            for chunk in chunks
        ]
        all_documents.extend(documents)

    print(f"[10] Total chunks to embed: {len(all_documents)}")

    if not all_documents:
        raise ValueError("No supported documents found to index.")

    # Ensure the target persistence directory exists before Chroma writes to it.
    os.makedirs(persist_dir, exist_ok=True)
    print("[11] Creating Chroma vector store...")
    start = time.time()

    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"[12] Chroma vector store built in {time.time() - start:.2f}s")
    print("[13] Vector store ready")

    return vectorstore
