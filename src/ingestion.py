import fitz  # PyMuPDF library used to extract text from PDF files
import pandas as pd
from typing import List
import os


def load_document(file_path: str) -> str:
    """
    Load document content based on file type.

    Supported formats:
    - PDF
    - TXT
    - CSV
    - XLSX / XLS

    Why this function matters:
    - The capstone requires support for multiple enterprise document formats.
    - All loaded content is converted into text so it can later be chunked,
      embedded, and stored in the vector database. [file:24]

    Args:
        file_path: Path to the input document.

    Returns:
        Extracted text content from the document.

    Raises:
        ValueError: If the file format is not supported.
    """
    # Extract the file extension so the loader knows how to process the file.
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        # Open the PDF using PyMuPDF.
        # Each page is read separately and then concatenated into one text string.
        # Page-by-page extraction helps preserve document flow reasonably well.
        doc = fitz.open(file_path)
        text = ""

        for page in doc:
            # get_text() extracts visible text from the page.
            # A newline is added between pages for better readability.
            text += page.get_text() + "\n"

        doc.close()
        return text.strip()

    elif ext == '.txt':
        # Read plain text files directly using UTF-8 encoding.
        # strip() removes leading and trailing whitespace.
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    elif ext in ['.csv', '.xlsx', '.xls']:
        # Structured tabular data is converted into markdown text.
        # This makes the content easier for the LLM to interpret in RAG prompts.
        if ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Convert table rows into a readable text format.
        # index=False prevents row numbers from being added unnecessarily.
        return df.to_markdown(index=False)

    else:
        # Unsupported files are rejected explicitly.
        # This makes error handling cleaner and easier to debug.
        raise ValueError(f"Unsupported format: {ext}. Use PDF/TXT/CSV/Excel.")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split large text into overlapping chunks for embedding and retrieval.

    Why chunking is needed:
    - Documents can be too large to embed or send to the LLM as one block.
    - Smaller chunks improve semantic retrieval accuracy.
    - Overlap helps preserve meaning across chunk boundaries. [file:24]

    Args:
        text: Full text extracted from a document.
        chunk_size: Maximum size of each chunk in characters.
        overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        A list of cleaned text chunks ready for embedding.
    """
    chunks = []
    start = 0

    while start < len(text):
        # Define the end position of the current chunk.
        end = start + chunk_size

        # Extract the chunk from the full text.
        chunk = text[start:end]
        chunks.append(chunk)

        # Move the window forward while preserving overlap.
        # Overlap ensures that important ideas split across boundaries
        # are still captured in neighboring chunks.
        start = end - overlap

    # Final cleanup:
    # - strip whitespace from each chunk
    # - remove very short chunks that may not be useful for retrieval
    return [c.strip() for c in chunks if len(c.strip()) > 50]
