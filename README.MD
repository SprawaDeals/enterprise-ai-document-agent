# 🤖 Enterprise AI Document Query Agent

<div align="center">

**Generative AI · Retrieval-Augmented Generation · ChromaDB · Agentic Workflow**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Core-1C3C3C)](https://www.langchain.com)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-6E44FF)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A capstone project for querying enterprise documents using RAG, semantic vector search, and an agent workflow consisting of Planner → Retriever → Reasoner → Validator.*

</div>

---

## ✨ Features

| Area | Description |
|------|-------------|
| **Document ingestion** | Supports PDF, TXT, CSV, and Excel files and converts them into text |
| **Chunking** | Splits large document text into overlapping chunks for better retrieval |
| **Embeddings** | Uses Hugging Face sentence-transformer embeddings |
| **Vector search** | Stores and searches chunk embeddings using ChromaDB |
| **RAG pipeline** | Combines retrieved chunks with user questions to generate grounded answers |
| **Agent workflow** | Uses Planner, Retriever, Reasoner, and Validator steps |
| **Safety** | Applies output validation to reduce hallucinations and unsafe answers |
| **UI** | Streamlit-based interface for ingestion and question answering |
| **Deployment** | Can run locally or inside Docker |

---

## 📋 Project Overview

This project is developed as part of the **Post Graduate Program in Generative AI and ML Capstone Project**. The objective is to build an AI-powered knowledge and decision support system that can process enterprise documents in multiple formats and answer natural language questions using LLMs, semantic retrieval, and agent-based reasoning.

The application supports:
- Multi-format document ingestion
- Text chunking for semantic search
- Embedding generation
- Vector-based retrieval using ChromaDB
- Retrieval-Augmented Generation (RAG)
- Agent-based planning, reasoning, and validation
- Safety and validation checks
- Streamlit-based user interface

---

## ⚙️ System Setup

### Prerequisites

Before running the application, install the following:

- Python 3.11
- pip
- Git
- Docker (optional)
- OpenAI API key

### Project Structure

```text
enterprise-ai-document-agent/
├── app.py
├── requirements.txt
├── .env
├── README.md
├── Dockerfile
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── embedding.py
│   ├── retriever.py
│   ├── agents.py
│   └── safety.py
├── data/
└── chroma_db/
