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
