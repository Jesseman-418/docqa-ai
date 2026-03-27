# DocQA-AI

A production-quality document question-answering system built on a Retrieval-Augmented Generation (RAG) pipeline. Upload PDF, TXT, or Markdown files and ask natural-language questions — the system retrieves the most relevant passages and presents them with relevance scores and source citations.

Runs entirely locally with no API keys required. Optionally enable generative answers via OpenAI.

```
                          Architecture
 ┌─────────────────────────────────────────────────────────┐
 │                     Streamlit UI                        │
 │  ┌──────────────┐  ┌─────────────────────────────────┐  │
 │  │   Sidebar     │  │         Chat Interface          │  │
 │  │  - Upload     │  │  User question                  │  │
 │  │  - Stats      │  │    │                            │  │
 │  │  - Settings   │  │    ▼                            │  │
 │  │  - Clear      │  │  Retrieved passages + scores    │  │
 │  └──────────────┘  └─────────────────────────────────┘  │
 └────────────┬────────────────────┬───────────────────────┘
              │ upload             │ query
              ▼                   ▼
 ┌────────────────────┐  ┌────────────────────┐
 │  Ingestion Layer   │  │  Retrieval Layer   │
 │  ┌──────────────┐  │  │  ┌──────────────┐  │
 │  │  PDF / TXT   │  │  │  │  Embedding    │  │
 │  │  Loader      │  │  │  │  Engine       │  │
 │  └──────┬───────┘  │  │  └──────┬───────┘  │
 │         ▼          │  │         ▼          │
 │  ┌──────────────┐  │  │  ┌──────────────┐  │
 │  │  Recursive   │  │  │  │  FAISS       │  │
 │  │  Chunker     │  │  │  │  VectorStore │  │
 │  └──────┬───────┘  │  │  └──────┬───────┘  │
 │         ▼          │  │         ▼          │
 │  ┌──────────────┐  │  │  ┌──────────────┐  │
 │  │  Sentence    │  │  │  │  Retriever   │  │
 │  │  Transformer │  │  │  │  (top-k +    │  │
 │  │  Embeddings  │  │  │  │   scoring)   │  │
 │  └──────────────┘  │  │  └──────────────┘  │
 └────────────────────┘  └────────┬───────────┘
                                  │
                                  ▼
                         ┌────────────────────┐
                         │    QA Chain         │
                         │  Extractive mode:   │
                         │    ranked passages  │
                         │  Generative mode:   │
                         │    LLM synthesis    │
                         └────────────────────┘
```

## Features

- **Multi-format ingestion** — PDF (PyPDF2), plain text, and Markdown
- **Recursive text chunking** — paragraph-aware splitting with configurable overlap for context preservation
- **Local embeddings** — sentence-transformers `all-MiniLM-L6-v2` (384-dim, runs on CPU, no API key)
- **FAISS vector store** — persistent index with save/load to disk
- **Similarity search** — cosine similarity with configurable top-k and score threshold
- **Extractive QA** — surfaces the most relevant passages with scores and source citations (no LLM needed)
- **Optional generative QA** — toggle on OpenAI-powered answers when an API key is available
- **Chat-style UI** — conversational interface with expandable source passages
- **Tested** — unit tests for chunking, vector storage, retrieval, and answer generation

## Tech Stack

| Component       | Technology                      |
| --------------- | ------------------------------- |
| Web UI          | Streamlit                       |
| Embeddings      | sentence-transformers           |
| Vector Store    | FAISS (faiss-cpu)               |
| PDF Parsing     | PyPDF2                          |
| LLM (optional)  | OpenAI API                     |
| Testing         | pytest                          |
| Language        | Python 3.12                     |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/docqa-ai.git
cd docqa-ai

# Install dependencies
make install

# Run the application
make run
```

The app opens at `http://localhost:8501`. Upload a document via the sidebar and start asking questions.

## Setup

### Requirements

- Python 3.12+
- ~500 MB disk space for the sentence-transformers model (downloaded on first run)

### Install

```bash
python3 -m pip install -r requirements.txt
```

### Run

```bash
python3 -m streamlit run app/main.py
```

### Test

```bash
python3 -m pytest tests/ -v
```

### Optional: Enable Generative Mode

Set your OpenAI API key to unlock LLM-powered answer synthesis:

```bash
export OPENAI_API_KEY="sk-..."
make run
```

Then toggle "Use LLM for answers" in the sidebar.

## Project Structure

```
docqa-ai/
├── app/
│   ├── main.py              # Streamlit application
│   ├── config.py            # Centralised settings (dataclasses)
│   ├── ingestion/
│   │   ├── loader.py        # PDF/TXT/MD file loaders
│   │   ├── chunker.py       # Recursive text chunking with overlap
│   │   └── embeddings.py    # Sentence-transformer embedding engine
│   ├── retrieval/
│   │   ├── vectorstore.py   # FAISS index management + persistence
│   │   └── retriever.py     # Similarity search with score filtering
│   ├── qa/
│   │   ├── chain.py         # Extractive + generative QA chain
│   │   └── prompts.py       # Prompt templates for LLM mode
│   └── utils/
│       └── text.py          # Unicode normalisation, whitespace cleaning
├── tests/
│   ├── test_chunker.py      # Chunking strategy tests
│   ├── test_retriever.py    # Vector store and retrieval tests
│   └── test_chain.py        # QA chain and prompt formatting tests
├── sample_docs/
│   └── sample.txt           # Sample document about RAG pipelines
├── requirements.txt
├── Makefile
└── .gitignore
```

## Design Decisions

1. **Extractive-first approach**: The default mode requires zero API keys. It retrieves and ranks passages, displaying them with relevance scores — this demonstrates the full RAG pipeline while being immediately usable.

2. **Recursive chunking**: Paragraph boundaries are preferred over fixed-size splits, preserving semantic coherence. Configurable overlap ensures context continuity across chunk boundaries.

3. **Normalised cosine similarity**: FAISS `IndexFlatIP` with L2-normalised vectors gives exact cosine similarity scores, making relevance scores interpretable (0-1 range).

4. **Singleton embedding model**: The sentence-transformer model is loaded once via `lru_cache` and shared across the session, avoiding repeated 500MB model loads.

5. **Persistent vector store**: The FAISS index and metadata are saved to disk after each upload, surviving Streamlit reruns and browser refreshes.

## License

MIT
