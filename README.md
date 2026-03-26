# RAG Document Search

**An agentic Retrieval-Augmented Generation (RAG) application for document-grounded question answering using Streamlit, LangGraph, FAISS, and Groq-powered reasoning.**

## Overview

RAG Document Search is a modular AI application that ingests documents, chunks and embeds them, stores them in a FAISS vector index, and answers user questions through an interactive Streamlit interface.

The project is structured around a clean `src/` package, a Streamlit frontend in `streamlit_app.py`, a LangGraph workflow for orchestration, Hugging Face sentence embeddings for retrieval, and a Groq-backed LLM for answer generation. It is designed as a practical portfolio project that demonstrates applied LLM engineering, document retrieval, workflow orchestration, and user-facing AI application design.

## Why This Project Matters

Large language models are most useful when they can answer questions grounded in real source material instead of relying only on parametric memory. This project demonstrates a practical Retrieval-Augmented Generation workflow that combines semantic search with LLM-based reasoning to produce more relevant and contextual answers.

It is especially valuable as a foundation for:

* internal knowledge assistants
* research document search tools
* enterprise Q&A systems
* educational AI applications
* portfolio-ready RAG engineering demonstrations

## Features

* Interactive **Streamlit UI** for document-grounded question answering
* Modular **document ingestion pipeline** for URLs, PDFs, and text files
* **Recursive text chunking** for retrieval-friendly context preparation
* **FAISS vector store** for fast semantic search
* **LangGraph-based workflow** for structured execution
* **ReAct-style answering agent**
* Retriever-first response generation
* Optional **Wikipedia fallback** for general knowledge lookup
* Expandable **source document viewer** in the UI
* Lightweight **recent search history**
* Clean, extensible architecture for future improvements

## Tech Stack

| Category               | Tools / Frameworks                                    |
| ---------------------- | ----------------------------------------------------- |
| Language               | Python                                                |
| UI                     | Streamlit                                             |
| Orchestration          | LangGraph                                             |
| LLM Framework          | LangChain                                             |
| LLM Provider           | Groq                                                  |
| Embeddings             | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database        | FAISS                                                 |
| Document Loading       | WebBaseLoader, PyPDFDirectoryLoader, TextLoader       |
| Environment Management | `python-dotenv`, `uv`, virtual environment            |
| Supporting Libraries   | `beautifulsoup4`, `requests`, `pypdf`, `wikipedia`    |

## Architecture / Workflow

```text
User Question
    ↓
Streamlit UI
    ↓
Initialize RAG System
    ├── Load configuration
    ├── Ingest source documents
    ├── Split documents into chunks
    ├── Generate embeddings
    ├── Build FAISS vector store
    └── Compile LangGraph workflow
                ↓
          Retriever Node
                ↓
       ReAct Answer Generator
       ├── Retriever Tool
       └── Wikipedia Tool
                ↓
          Final Answer + Sources
```

### Workflow Summary

1. The application initializes the RAG system from configuration.
2. Source documents are loaded and split into chunks.
3. Chunks are embedded and indexed in FAISS.
4. A LangGraph workflow is compiled with retrieval and answer-generation stages.
5. When a user submits a query, relevant chunks are retrieved.
6. A ReAct-style agent generates the final answer using indexed context, with Wikipedia available as a fallback tool.
7. The UI displays the response, source snippets, and recent query history.

## Dataset / Knowledge Sources

This repository is currently configured around a lightweight document corpus rather than a traditional ML training dataset.

### Included Sources

* Default web sources are defined in the configuration module
* The `data/` directory includes:

  * `attention.pdf`
  * `url.txt`

### Current Retrieval Context

The application uses repository data assets and configured web sources as part of its ingestion pipeline. This makes the project suitable for demonstrating document-grounded inference without requiring a separate training dataset.

## Model Details

### LLM

* Provider: **Groq**
* Configured model: **`openai/gpt-oss-20b`**

### Embedding Model

* **`sentence-transformers/all-MiniLM-L6-v2`**

### Retrieval Layer

* **FAISS** for vector similarity search

### Reasoning Layer

* **LangGraph** state workflow
* **ReAct-style tool-using agent**
* Retriever-first answering with optional Wikipedia fallback

## Training Process

This project does **not** train a custom model. It is an **inference-time RAG system** focused on:

* document ingestion
* chunking
* embedding generation
* semantic retrieval
* grounded answer generation

That makes it suitable for demonstrating applied LLM engineering and RAG system design rather than model fine-tuning.

## Evaluation Metrics

The current repository emphasizes interactive retrieval and inference through the UI rather than an automated offline benchmark suite.

Recommended evaluation metrics for future improvement include:

* **Precision@K**
* **Recall@K**
* **MRR / nDCG**
* **Answer Faithfulness**
* **Answer Relevance**
* **Latency per Query**

## Inference

At inference time, the user submits a natural-language question through the Streamlit interface. The system retrieves relevant context from the indexed document corpus, routes the question through the LangGraph workflow, and generates a final answer using the configured LLM and available tools. Source passages are exposed in the UI to improve transparency and trust.

## Core Modules

| Module                                         | Responsibility                                                                |
| ---------------------------------------------- | ----------------------------------------------------------------------------- |
| `streamlit_app.py`                             | Streamlit frontend, app state, initialization, query flow                     |
| `src/config/config.py`                         | Environment variables, model config, chunking config, default source URLs     |
| `src/document_ingestion/document_processor.py` | Loads documents from URLs, PDFs, and text files, then splits them into chunks |
| `src/vectorstore/vectorstore.py`               | Embedding creation and FAISS vector store setup                               |
| `src/graph_builder/graph_builder.py`           | Builds and runs the LangGraph workflow                                        |
| `src/nodes/reactnode.py`                       | Retrieval node plus ReAct-based answer generation with tools                  |
| `src/state/rag_state.py`                       | Shared state object for workflow execution                                    |
| `main.py`                                      | Minimal package entry stub                                                    |

## Project Structure

```text
RAG-Document-Search/
├── data/
│   ├── attention.pdf
│   └── url.txt
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── document_ingestion/
│   │   ├── __init__.py
│   │   └── document_processor.py
│   ├── graph_builder/
│   │   ├── __init__.py
│   │   └── graph_builder.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   └── reactnode.py
│   ├── state/
│   │   ├── __init__.py
│   │   └── rag_state.py
│   └── vectorstore/
│       ├── __init__.py
│       └── vectorstore.py
├── .env
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── streamlit_app.py
└── uv.lock
```

## Installation

### Prerequisites

* Python **3.13+**
* A valid **Groq API key**
* `pip` or `uv`

### Option 1: Using `pip`

```bash
git clone https://github.com/SAKIB0004/RAG-Document-Search.git
cd RAG-Document-Search

python -m venv .venv
```

#### Activate the virtual environment

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

### Option 2: Using `uv`

```bash
git clone https://github.com/SAKIB0004/RAG-Document-Search.git
cd RAG-Document-Search
uv sync
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=optional
```

## Usage

### Run the Application

```bash
streamlit run streamlit_app.py
```

### Example Queries

```text
What is an AI agent?
Summarize the key idea of diffusion video generation.
Explain the main concept discussed in the indexed documents.
```

### User Experience

* Enter a question in the search box
* Receive a generated answer
* Expand source snippets for supporting context
* Review recent searches and response times

## Example Output / Results

```text
Question:
What is an AI agent?

Expected Behavior:
The system retrieves relevant chunks from the indexed corpus,
grounds the answer in those passages,
and returns a concise response in the UI along with source context.
```

```text
Question:
Summarize the indexed content.

Expected Behavior:
The app synthesizes a retrieval-backed summary
and surfaces the supporting passages under the source document panel.
```



## Deployment Notes

This project is currently set up for local development and demonstration. For production or public deployment, consider the following:

* externalize configuration and secrets management
* persist FAISS indexes instead of rebuilding on each run
* containerize with Docker
* add health checks and logging
* deploy the Streamlit frontend separately from backend services if the architecture expands
* use a managed vector database for larger-scale corpora

## Roadmap

* [ ] Add file upload support in the UI
* [ ] Support dynamic source selection instead of fixed default URLs
* [ ] Add persistent vector index storage
* [ ] Introduce automated evaluation metrics for retrieval quality
* [ ] Add conversation memory for multi-turn QA
* [ ] Improve prompt engineering and grounding safeguards
* [ ] Containerize the application with Docker
* [ ] Add deployment instructions for Streamlit Community Cloud or other hosting targets
* [ ] Add unit and integration tests
* [ ] Expand observability with logging and tracing


---
