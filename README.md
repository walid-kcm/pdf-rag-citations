# PDF RAG with Citations (LangChain + ChromaDB + Groq)

A lightweight Retrieval-Augmented Generation (RAG) app to query scientific PDF documents.  
Built with LangChain, a persistent Chroma vector store, and Groq for fast inference. Answers include page-level citations.

## Features
- Loads PDFs from `data/documents/`
- Chunking with overlap to preserve context
- Persistent local vector store (ChromaDB)
- Streamlit web UI
- Answers with sources (page numbers)
- Basic caching to avoid rebuilding embeddings unnecessarily

## Tech stack
- LangChain (RAG pipeline)
- ChromaDB (local vector store)
- Groq API (LLM inference)
- Sentence-Transformers embeddings (`all-MiniLM-L6-v2`)
- PyPDF2 (PDF parsing)
- Streamlit (UI)
- python-dotenv (env variables)

## Requirements
- Python 3.8+
- A Groq API key

## Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
