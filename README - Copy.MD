# Enterprise AI Agent: Generative AI Document Query System

## Overview
Autonomous AI agents enable natural language queries over enterprise docs (PDF/TXT/CSV/Excel) via RAG + agentic reasoning. Built for PGP Generative AI/ML Capstone.[file:24]

## Architecture
- **Ingestion**: Multi-format loader → chunking.
- **Vector Store**: ChromaDB + SentenceTransformers.
- **Agents**: CrewAI (Planner/Retriever/Reasoner/Validator).
- **RAG/LLM**: LangChain + OpenAI.
- **UI**: Streamlit.
- **Safety**: Pydantic guardrails.

ASCII Diagram:
Enterprise AI Document Query Agent
Generative AI + RAG + Agentic Workflow for Enterprise Documents

1. System Setup
1.1 Prerequisites
Python 3.11 installed.

Git and Docker (optional, for containerized deployment).

OpenAI API key (stored in a local .env file as OPENAI_API_KEY).

Local folder ./data containing enterprise documents (PDF, TXT, CSV, Excel).
​

1.2 Project Structure
text
enterprise-ai-agent/
├── app.py                     # Streamlit UI and main application flow
├── requirements.txt           # Python dependencies
├── .env                       # OpenAI API key and other config (not committed)
├── src/
│   ├── config.py              # Central configuration (Pydantic Settings)
│   ├── ingestion.py           # Document loading and text chunking
│   ├── embedding.py           # Embeddings + Chroma vector store
│   ├── retriever.py           # Semantic retrieval abstraction
│   ├── agents.py              # Planner, Reasoner, Validator agent steps
│   └── safety.py              # Output validation and guardrails
├── data/                      # Input documents (PDF/TXT/CSV/Excel)
└── chroma_db/                 # Persisted Chroma vector store
1.3 Installation Steps
Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Configure environment variables in .env:

text
OPENAI_API_KEY=sk-...
Prepare documents:

Place enterprise documents into ./data (e.g., DAMA DMBOK 2nd Edition.pdf).
​

(Optional but recommended) Build the vector store once from CLI:

bash
python -c "from src.embedding import build_vector_store; build_vector_store('./data')"
2. System Architecture
2.1 High-Level Workflow
The application implements a full Generative AI and Agentic AI workflow over enterprise documents:
​

text
User (Streamlit UI)
      │
      ▼
Natural Language Query
      │
      ▼
Agentic Pipeline (Planner → Retriever → Reasoner → Validator)
      │
      ▼
RAG Layer (Retrieval-Augmented Generation)
      │
      ├─► Chroma Vector Store (Embeddings)
      │       ▲
      │       └─ Ingestion & Chunking of Documents (PDF/TXT/CSV/XLSX)
      ▼
Validated Answer (with sources)
2.2 Component Diagram (Text Form)
text
+------------------------+       +----------------------+
|        app.py          |       |      config.py       |
|  - Streamlit Frontend  |<----->|  - Settings (API key,|
|  - Session State       |       |    models, paths)   |
+-----------+------------+       +----------+-----------+
            |                                |
            v                                v
+-----------+------------+       +----------+-----------+
|      agents.py         |       |   ingestion.py       |
|  - Planner agent       |       | - load_document()    |
|  - Reasoner agent      |       | - chunk_text()       |
|  - Validator agent     |       +----------+-----------+
+-----------+------------+                  |
            |                               v
            v                      +--------+----------+
+-----------+------------+         |   embedding.py    |
|    retriever.py        |<------->| - HuggingFace     |
| - EnterpriseRetriever  |         |   Embeddings      |
|   (semantic search)    |         | - Chroma Vector   |
+-----------+------------+         |   Store           |
            |                      +--------+----------+
            v                               |
+-----------+------------+                  v
|    chroma_db/          |          Persisted Embeddings
+------------------------+
2.3 Key Technologies
Streamlit for the user interface and interaction layer.

PyMuPDF (fitz) and pandas for multi-format document ingestion.

HuggingFaceEmbeddings (via langchain_huggingface) + Chroma for vector-based knowledge store.

LangChain OpenAI client for LLM interaction.

Pydantic Settings for centralized configuration.

Custom agentic pipeline implemented in agents.py rather than relying on heavy external agent frameworks, to keep the solution robust and focused on the capstone goals.
​

3. Detailed Component Description
3.1 Configuration (src/config.py)
Central location for:

openai_api_key

embedding_model

llm_model

chroma_persist_dir

chunk_size, chunk_overlap

top_k, max_new_tokens

Uses BaseSettings from pydantic_settings to load values from .env, making the project easy to move between environments (local, cloud, container).
​

3.2 Ingestion and Chunking (src/ingestion.py)
load_document(file_path: str) -> str

Handles PDF, TXT, CSV, XLS/XLSX as required by the capstone.
​

PDF: PyMuPDF extracts text page by page.

TXT: UTF‑8 text read directly.

CSV/Excel: Converted to markdown tables for LLM-friendly context.

chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]

Splits large documents into overlapping chunks.

Overlap preserves semantic continuity across boundaries.

Filters out trivial chunks (< 50 characters).

This stage fulfills Tasks 3 and 4: document ingestion and preparation for semantic search.
​

3.3 Embeddings and Vector Store (src/embedding.py)
Initializes HuggingFaceEmbeddings using a SentenceTransformers model.

build_vector_store(docs_dir: str) -> Chroma:

Validates docs_dir exists.

Attempts to load existing Chroma DB from settings.chroma_persist_dir.

If not available, reads each supported file:

load_document() → text.

chunk_text() → chunks.

Wraps each chunk as Document(page_content, metadata={"source": filename}).

Creates or updates a Chroma vector store with persistent storage.

This stage implements Tasks 5 and 6: vector-based knowledge store and intelligent document retrieval foundation.
​

3.4 Retriever (src/retriever.py)
EnterpriseRetriever encapsulates Chroma access:

.as_retriever() returns a LangChain-style retriever (if needed).

.get_relevant_documents(query: str) -> List[Document] uses similarity_search with top‑k chunks.

This class provides a clear boundary between raw storage and the agentic reasoning pipeline.

3.5 Agent Roles and Workflow (src/agents.py)
The agentic workflow is implemented as three logical “agents”, each as a dedicated function using the same LLM client:

Planner Agent – _planner_step

Role: Query Planning Agent.

Input: User question.

Output: Short plan including intent, key topics, and evidence to retrieve.

Purpose: Makes retrieval more focused and explainable.

Reasoner Agent – _reasoner_step

Role: Reasoning Agent in an Enterprise RAG System.

Inputs: User question, retrieval plan, and retrieved context (chunks + sources).

Behavior:

Instructed to use only retrieved context.

Must be concise, accurate, and clearly mention sources.

Purpose: Generate the first grounded draft answer.

Validator Agent – _validator_step

Role: Validation Agent.

Inputs: Original question, draft answer, and retrieved context.

Behavior:

Checks if the draft is grounded in context.

Evaluates hallucination risk and source usage.

Optionally rewrites the answer to be safer and more accurate.

Purpose: Provide reliability and “second opinion” before the answer is shown.

Orchestrator – run_agentic_pipeline

Pipeline:

Planner analyzes the query.

Retriever fetches top‑k relevant chunks from Chroma.

Reasoner produces a draft answer from the context.

Validator reviews and refines the answer.

Returns the final, validated answer to the UI.

This satisfies Tasks 7 and 8: Retrieval-Augmented Generation and agent-based reasoning.
​

3.6 Safety and Guardrails (src/safety.py)
validate_output(response: str) -> str:

Flags speculative language (e.g., “guess”, “probably”) indicating potential hallucination.

Blocks responses containing sensitive terms like "password", "SSN", or "confidential".

Requires “source” to appear in longer answers to encourage grounded responses.

Returns either the original response or a user-friendly validation failure message.

This implements Task 9: reliability and safety controls.
​

3.7 User Interface (app.py)
Sidebar:

Documents folder input (default ./data).

“Build/Update Vector Store” button to trigger ingestion and embedding.

Visual feedback on vector store readiness.

Main Area:

Title and short description of the application.

Chat-style interface:

Maintains st.session_state.messages for persistent conversation.

Accepts natural language questions.

Displays both user queries and AI responses.

Core flow on user query:

User enters a question in the chat input.

run_agentic_pipeline(prompt, retriever) is called.

Response is passed through validate_output.

Validated answer is appended to chat history and displayed.

Footer:

Lists capstone tasks demonstrated.

Shows a Docker deployment command as example.

4. Deployment Steps
4.1 Local Development Run
Ensure virtual environment is active and requirements installed.

Confirm .env contains a valid OPENAI_API_KEY.

Start the Streamlit app:

bash
streamlit run app.py
Open http://localhost:8501 in a browser.

In the sidebar:

Confirm Documents folder is ./data.

Click “Build/Update Vector Store”.

Ask questions like:

“What is Data Governance? Explain in simple layman terms with examples.”

“Summarize the key roles in DAMA DMBOK data governance.”

4.2 Docker-Based Deployment
Build the image:

bash
docker build -t ai-agent .
Run the container (mounting data and DB if desired):

bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  ai-agent
Access the app via http://localhost:8501 (or the host IP if deployed remotely).

You can deploy similarly to any container-friendly platform (e.g., Docker on a VM, Azure Container Apps, etc.), noting that the app is stateless apart from the mounted chroma_db and data folders.

5. Limitations and Challenges
5.1 Technical Limitations
Model Dependency and Costs

The system currently depends on an external LLM (OpenAI) for planning, reasoning, and validation, which introduces latency and token costs.

For very large workloads or strict data residency requirements, an on-prem or open-source LLM would be more suitable.

Embedding Model and Language Coverage

The chosen MiniLM-based embedding model works well for English but may not perform optimally on multi-lingual or domain-specific jargon.

Hybrid search (combining lexical + semantic) is not implemented yet.

Vector Store Refresh

Once the Chroma DB exists, build_vector_store loads it and returns early; newly added documents require an explicit rebuild or index refresh.

There is no explicit document deletion or per-document index management.

Basic Guardrails

Safety checks focus on speculative language, a small set of sensitive keywords, and source mention.

More advanced techniques such as structured fact-checking, content classification, or policy-driven filters are not implemented.

Single-User Session

The app is designed for a single-user context using Streamlit session state.

Multi-user concurrency and user-level isolation (e.g., per-tenant indices) are out of scope.

5.2 Development Challenges and How They Were Addressed
Library Version Mismatch

LangChain’s rapid evolution caused import issues (e.g., langchain.docstore, HuggingFaceEmbeddings deprecations, Chroma persist() removal).

Resolved by:

Using langchain_core.documents.Document.

Moving to langchain_huggingface.HuggingFaceEmbeddings.

Relying on Chroma’s persist_directory instead of persist().

Pydantic v2 Migration

BaseSettings was moved to pydantic-settings, causing PydanticImportError.

Resolved by importing from pydantic_settings and updating config patterns accordingly.

Keeping Agentic Design Simple and Stable

Early attempts to use advanced agent frameworks introduced additional dependencies and import errors.

The final design uses a lightweight, function-based agent pipeline (planner → reasoner → validator) on top of LangChain’s LLM client, which is easier to maintain and explain in an academic context.

Balancing Chunk Size and Overlap

Too small chunks lost context; too large chunks hurt retrieval precision and token usage.

Empirically tuned chunk_size = 1000 and overlap = 200, which worked well for typical enterprise PDFs like DAMA DMBOK-style documents.

User Experience in Streamlit

Managing state across reruns and preventing uninitialized keys caused early errors.

Addressed by explicit initialization of st.session_state.vectorstore, st.session_state.retriever, and st.session_state.messages at the top of app.py.