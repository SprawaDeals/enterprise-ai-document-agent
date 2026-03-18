import os
import uuid
import shutil
from pathlib import Path

import streamlit as st

from src.embedding import build_vector_store
from src.retriever import EnterpriseRetriever
from src.agents import run_agentic_pipeline
from src.config import settings


BASE_DIR = Path(".")
UPLOAD_ROOT = BASE_DIR / "data_sessions"
VECTOR_ROOT = BASE_DIR / "vectorstores"

# Create base folders once so uploads and vector stores have dedicated locations.
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
VECTOR_ROOT.mkdir(parents=True, exist_ok=True)


def init_session_state():
    """
    Initialize session-scoped values used across Streamlit reruns.
    """
    # Each browser session gets its own upload and vector store paths.
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex

    if "upload_dir" not in st.session_state:
        st.session_state["upload_dir"] = str(UPLOAD_ROOT / f"upload_{st.session_state['session_id']}")

    if "persist_dir" not in st.session_state:
        st.session_state["persist_dir"] = str(VECTOR_ROOT / f"db_{st.session_state['session_id']}")

    # Retriever is populated only after indexing succeeds.
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    if "indexed_files" not in st.session_state:
        st.session_state["indexed_files"] = []

    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = ""


def ensure_clean_dir(path: str):
    """
    Create the target directory if it does not already exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def clear_directory_contents(path: str):
    """
    Remove files/subfolders from a directory before rebuilding uploads or indexes.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return

    for item in path_obj.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception:
            # Ignore cleanup issues so a single locked file does not crash the UI.
            pass


def save_uploaded_files(uploaded_files, target_dir: str) -> list[str]:
    """
    Save uploaded files into a session-specific folder using unique filenames.
    """
    ensure_clean_dir(target_dir)
    saved_files = []

    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name
        safe_name = original_name.replace(" ", "_")
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        file_path = Path(target_dir) / unique_name

        # Save uploaded file bytes to disk so the ingestion pipeline can read them.
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_files.append(str(file_path))

    return saved_files


def initialize_retriever(data_dir: str, persist_dir: str):
    """
    Build the vector store and wrap it in the custom retriever class.
    """
    vectorstore = build_vector_store(data_dir, persist_dir)
    return EnterpriseRetriever(vectorstore, k=settings.retrieval_k)


# Initialize session variables before rendering widgets.
init_session_state()

st.set_page_config(
    page_title="Enterprise AI Document Query Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS is used to match the desired capstone UI style.
st.markdown(
    """
    <style>
        .main {
            background-color: #ffffff;
        }

        section[data-testid="stSidebar"] {
            background-color: #f7f8fc;
            border-right: 1px solid #e6e8ef;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            font-size: 0.95rem;
            color: #6b7280;
            margin-bottom: 1rem;
        }

        .info-box {
            background-color: #eef5ff;
            border: 1px solid #d6e8ff;
            color: #315b9a;
            padding: 0.85rem 1rem;
            border-radius: 10px;
            margin: 1rem 0 1.5rem 0;
            font-size: 0.95rem;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #eceff5;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            margin-top: 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }

        .success-chip {
            display: inline-block;
            background: #ecfdf3;
            color: #027a48;
            border: 1px solid #abefc6;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.8rem;
            margin-top: 0.4rem;
        }

        .feature-list {
            line-height: 1.8;
            color: #374151;
            font-size: 0.96rem;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            border: none !important;
        }

        section[data-testid="stSidebar"] div.stButton > button {
            background-color: #dc2626 !important;
            border: 1px solid #dc2626 !important;
        }

        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #b91c1c !important;
            border: 1px solid #b91c1c !important;
        }

        div[data-testid="stFormSubmitButton"] > button {
            background-color: #2563eb !important;
            border: 1px solid #2563eb !important;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #1d4ed8 !important;
            border: 1px solid #1d4ed8 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 📁 Knowledge Base Setup")

    # Upload supports the document types required for the capstone use case.
    uploaded_files = st.file_uploader(
        "Documents Folder",
        type=["pdf", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if st.button("Build / Update Vector Store", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            try:
                # Replace previous session files/index to keep the knowledge base current.
                clear_directory_contents(st.session_state["upload_dir"])
                clear_directory_contents(st.session_state["persist_dir"])

                saved_paths = save_uploaded_files(uploaded_files, st.session_state["upload_dir"])

                # Build embeddings and persist the vector store for this session.
                with st.spinner("Building vector store..."):
                    retriever = initialize_retriever(
                        st.session_state["upload_dir"],
                        st.session_state["persist_dir"]
                    )

                st.session_state["retriever"] = retriever
                st.session_state["indexed_files"] = [Path(p).name for p in saved_paths]
                st.success(f"Indexed {len(saved_paths)} document(s).")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if st.session_state["indexed_files"]:
        st.markdown("---")
        st.markdown("**Indexed Files**")
        for file_name in st.session_state["indexed_files"]:
            st.caption(file_name)

st.markdown('<div class="hero-title">🤖 Enterprise AI Document Query Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Generative AI + RAG + Autonomous Agents for enterprise documents (PDF/TXT/CSV/Excel)</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-box">💡 Step 1: Use the sidebar to build the vector store from your documents.</div>',
    unsafe_allow_html=True
)

if st.session_state["retriever"] is None:
    # Before indexing, show a project summary instead of the ready-state indicator.
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Capstone Features Demonstrated")
    st.markdown(
        """
<div class="feature-list">
- Multi-format document ingestion (Task 3)<br>
- Semantic vector search (Tasks 4-6)<br>
- RAG + agentic workflow (Tasks 7-8)<br>
- Safety validation and guardrails (Task 9)<br>
- Deployable Streamlit user interface (Task 10)
</div>
        """,
        unsafe_allow_html=True
    )
    st.caption('Deployment Example: `docker build -t ai-agent . && docker run -p 8501:8501 ai-agent`')
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="success-chip">Vector store ready</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("#### Ask Questions")

# Use a form so pressing Enter submits reliably and the Ask button stays visible.
with st.form("question_form", clear_on_submit=False):
    user_query = st.text_input(
        "Enter your question",
        placeholder="Example: What is the revenue for Q1 2023 according to the financial report?",
        label_visibility="collapsed"
    )
    ask_clicked = st.form_submit_button("Ask")

if ask_clicked:
    if st.session_state["retriever"] is None:
        st.warning("Please build the vector store first.")
    elif not user_query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            retriever = st.session_state["retriever"]

            # Execute the agentic RAG workflow using the indexed enterprise documents.
            with st.spinner("Running agent..."):
                answer = run_agentic_pipeline(user_query.strip(), retriever)

            st.session_state["last_answer"] = answer

        except Exception as e:
            st.error(f"Question answering failed: {e}")

if st.session_state["last_answer"]:
    st.markdown("#### Response")
    st.write(st.session_state["last_answer"])

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state["retriever"] is not None:
    # Show session-specific paths to make debugging and demos easier.
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Current Session")
    st.write(f"**Session ID:** {st.session_state['session_id']}")
    st.write(f"**Upload folder:** `{st.session_state['upload_dir']}`")
    st.write(f"**Vector store:** `{st.session_state['persist_dir']}`")
    st.markdown("</div>", unsafe_allow_html=True)
