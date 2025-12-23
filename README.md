# Document Q&A Assistant - Local RAG Pipeline

A local RAG (Retrieval-Augmented Generation) application that allows users to upload PDF documents and ask questions about them. The system runs entirely locally using Ollama for LLM inference, ensuring data privacy.

## Prerequisites

1. **Python 3.9+** installed on your system
2. **Ollama** installed and running
   - Download from: https://ollama.com/
   - Install and start the Ollama service
3. **Mistral Model** pulled in Ollama:
   ```bash
   ollama pull mistral
   ```

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser
3. Upload a PDF document using the file uploader
4. Wait for the document to be processed (chunked, embedded, and indexed)
5. Enter your question in the text input field
6. View the answer generated from the document

## Architecture

### Components

1. **Ingest** (`rag/loader.py`): Loads PDF files using LangChain's PyPDFLoader
2. **Process** (`rag/splitter.py`): Splits documents into manageable chunks using RecursiveCharacterTextSplitter
   - Chunk size: 500 characters
   - Chunk overlap: 50 characters
   - **Rationale**: 500 characters provides a good balance between context preservation and retrieval precision. The 50-character overlap ensures continuity between chunks, preventing important information from being split across boundaries.
3. **Embed** (`rag/embeddings.py`): Converts chunks into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` and stores them in FAISS
4. **Retrieve** (`rag/retriever.py`): Finds top 4 most relevant chunks using similarity search
5. **Generate** (`rag/generator.py`): Uses Ollama with Mistral model to generate answers strictly from retrieved context

### Key Design Decisions

- **Embedding Model**: `all-MiniLM-L6-v2` - Lightweight, fast, and effective for semantic search
- **Vector Database**: FAISS - Fast, local, and memory-efficient
- **LLM**: Mistral via Ollama - Open-source, local, and privacy-preserving
- **Chunk Strategy**: RecursiveCharacterTextSplitter with 500/50 split - Balances context and granularity

### Critical Requirement Compliance

The system is designed to answer **ONLY** from the provided document:
- If the answer exists in the document → Provides the answer
- If the answer is not found → Explicitly states: "I cannot find the answer to that question in the provided document."
- The prompt explicitly instructs the LLM to not hallucinate or use outside knowledge

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── rag/                  # RAG pipeline modules
│   ├── __init__.py
│   ├── loader.py         # PDF loading
│   ├── splitter.py       # Document chunking
│   ├── embeddings.py      # Vector embeddings & FAISS
│   ├── retriever.py      # Context retrieval
│   └── generator.py      # Answer generation
└── data/                 # Generated data (FAISS index)
    └── faiss_index/
```

## Notes

- The FAISS index is saved locally in `data/faiss_index/` for persistence
- Each new document upload creates a new index (replaces the previous one)
- Ensure Ollama is running before starting the application
- The first run may take longer as the embedding model downloads

