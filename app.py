import streamlit as st
import os
from src.embedding import build_vector_store
from src.agents import run_agentic_pipeline
from src.retriever import EnterpriseRetriever
from src.config import settings
from src.safety import validate_output


# -----------------------------
# Session State Initialization
# -----------------------------
# Streamlit reruns the script on every interaction.
# These checks ensure that important objects remain available
# across reruns, such as the vector store, retriever, and chat history.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Page Configuration
# -----------------------------
# Set the page title and layout for a cleaner UI.
st.set_page_config(page_title="Enterprise AI Agent", layout="wide")


# -----------------------------
# Main App Title
# -----------------------------
st.title("🤖 Enterprise AI Document Query Agent")
st.markdown(
    "**Generative AI + RAG + Autonomous Agents** for enterprise documents (PDF/TXT/CSV/Excel)"
)


# -----------------------------
# Sidebar: Knowledge Base Setup
# -----------------------------
# This section supports:
# - Task 1: project/application setup
# - Task 2: user interaction layer
st.sidebar.header("📁 Knowledge Base Setup")

# User provides the folder path where documents are stored.
docs_dir = st.sidebar.text_input(
    "Documents folder",
    value="./data",
    help="Add your PDF, TXT, CSV, or Excel files here."
)

# Button to build or refresh the vector store.
ingest_btn = st.sidebar.button("🔄 Build/Update Vector Store", type="primary")


if ingest_btn:
    with st.spinner("Processing documents → chunking → embeddings → ChromaDB..."):
        try:
            # Build or load the vector store from the selected folder.
            st.session_state.vectorstore = build_vector_store(docs_dir)

            # Wrap the vector store in the retriever class used by the agent pipeline.
            st.session_state.retriever = EnterpriseRetriever(st.session_state.vectorstore)

            # Show a success message in the sidebar.
            st.sidebar.success(
                f"✅ Vector store ready!\n📊 {len(os.listdir(docs_dir))} files available"
            )

        except Exception as e:
            # If anything goes wrong during ingestion, show a readable error.
            st.sidebar.error(f"❌ Error while building vector store: {str(e)}")


# -----------------------------
# API Key Validation
# -----------------------------
# The app needs an OpenAI API key for the LLM-powered planning,
# reasoning, and validation steps.
if not settings.openai_api_key:
    st.sidebar.warning("⚠️ Add `OPENAI_API_KEY=sk-...` to the `.env` file")
    st.stop()


# -----------------------------
# Prompt User to Ingest First
# -----------------------------
# If the vector store is not yet loaded, guide the user to ingest documents first.
if st.session_state.vectorstore is None:
    st.info("👆 Step 1: Use the sidebar to build the vector store from your documents.")


# -----------------------------
# Chat History Display
# -----------------------------
# Display all prior chat messages stored in session state.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User Query Input
# -----------------------------
# The chat input is enabled only after the retriever is available.
if st.session_state.retriever is not None:
    prompt = st.chat_input("Ask about your enterprise documents...")

    if prompt:
        # Save and display the user's question.
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query through the agentic RAG pipeline.
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agents working: Planning → Retrieving → Reasoning → Validating..."):
                try:
                    # Generate response using the multi-step agentic workflow.
                    raw_response = run_agentic_pipeline(prompt, st.session_state.retriever)

                    # Apply safety and grounding validation before displaying output.
                    final_response = validate_output(raw_response)

                    # Show validated answer to the user.
                    st.markdown(final_response)

                    # Store assistant response in chat history.
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_response}
                    )

                except Exception as e:
                    # Friendly error message if the agent workflow fails.
                    error_msg = (
                        f"❌ Agent error: {str(e)}\n\n"
                        f"**Debug hint:** Check console logs, API key, and vector store."
                    )
                    st.markdown(error_msg)
                    st.error(error_msg)


# -----------------------------
# Footer: Capstone Coverage
# -----------------------------
# This section helps demonstrate alignment with the capstone requirements.
st.markdown("---")
st.markdown(
    """
**Capstone Features Demonstrated**:
- ✅ Multi-format document ingestion (Task 3)
- ✅ Semantic vector search (Tasks 4-6)
- ✅ RAG + agentic workflow (Tasks 7-8)
- ✅ Safety validation and guardrails (Task 9)
- ✅ Deployable Streamlit user interface (Task 10)

**Deployment Example**: `docker build -t ai-agent . && docker run -p 8501:8501 ai-agent`
"""
)
