"""DocQA-AI — Streamlit web application."""

from __future__ import annotations

import os
import logging
from pathlib import Path

import streamlit as st

from app.config import settings
from app.ingestion.loader import load_document
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.embeddings import EmbeddingEngine
from app.retrieval.vectorstore import VectorStore
from app.retrieval.retriever import Retriever
from app.qa.chain import QAChain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocQA-AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    # Try to load a persisted index; otherwise start fresh.
    try:
        st.session_state.vector_store = VectorStore.load()
        logger.info("Loaded persisted vector store.")
    except FileNotFoundError:
        st.session_state.vector_store = VectorStore()

if "embedding_engine" not in st.session_state:
    st.session_state.embedding_engine = EmbeddingEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_qa_chain() -> QAChain:
    retriever = Retriever(st.session_state.vector_store, st.session_state.embedding_engine)
    return QAChain(retriever)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("DocQA-AI")
    st.caption("AI-powered document question answering")

    st.divider()

    # ---- File upload ----
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            chunker = RecursiveChunker()
            engine = st.session_state.embedding_engine
            store: VectorStore = st.session_state.vector_store

            for uploaded in uploaded_files:
                # Skip if already indexed
                if uploaded.name in store.get_sources():
                    st.info(f"'{uploaded.name}' already indexed — skipped.")
                    continue

                raw_bytes = uploaded.read()
                doc = load_document(uploaded.name, raw_bytes=raw_bytes)
                chunks = chunker.chunk(doc.text, metadata=doc.metadata)
                embeddings = engine.embed_texts([c.text for c in chunks])
                store.add(embeddings, chunks)
                st.success(f"Indexed '{uploaded.name}' — {len(chunks)} chunks")

            store.save()

    st.divider()

    # ---- Index stats ----
    st.subheader("Index Stats")
    store = st.session_state.vector_store
    col1, col2 = st.columns(2)
    col1.metric("Vectors", store.size)
    col2.metric("Documents", len(store.get_sources()))

    if store.get_sources():
        with st.expander("Indexed documents"):
            for src in store.get_sources():
                st.markdown(f"- `{src}`")

    st.divider()

    # ---- Settings ----
    st.subheader("Settings")
    top_k = st.slider("Results to retrieve", min_value=1, max_value=20, value=5)

    use_llm = False
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    use_llm = st.toggle(
        "Use LLM for answers (requires OPENAI_API_KEY)",
        value=False,
        disabled=not has_key,
        help="Enable generative answers via OpenAI. Set the OPENAI_API_KEY env var to activate.",
    )

    st.divider()

    # ---- Clear index ----
    if st.button("Clear Index", type="secondary", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.chat_history = []
        # Remove persisted files
        idx_dir = Path(settings.index_dir)
        for f in idx_dir.glob("*"):
            f.unlink()
        st.rerun()

# ---------------------------------------------------------------------------
# Main area — Chat-style Q&A
# ---------------------------------------------------------------------------
st.header("Ask your documents a question")

if store.size == 0:
    st.info("Upload one or more documents using the sidebar to get started.")
else:
    # Render chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("passages"):
                with st.expander(f"Source passages ({len(entry['passages'])})"):
                    for i, p in enumerate(entry["passages"], 1):
                        src = p["metadata"].get("source", "unknown")
                        st.markdown(f"**Passage {i}** — `{src}` (score: {p['score']:.3f})")
                        st.code(p["text"], language=None)

    # Input
    question = st.chat_input("Type your question here...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                chain = get_qa_chain()
                result = chain.answer(question, top_k=top_k, use_llm=use_llm)

            st.markdown(result.answer_text)

            if result.passages:
                with st.expander(f"Source passages ({len(result.passages)})"):
                    for i, p in enumerate(result.passages, 1):
                        src = p["metadata"].get("source", "unknown")
                        st.markdown(f"**Passage {i}** — `{src}` (score: {p['score']:.3f})")
                        st.code(p["text"], language=None)

            if result.mode == "generative":
                st.caption("Answer generated by LLM using retrieved context.")
            else:
                st.caption("Showing most relevant passages (extractive mode).")

        st.session_state.chat_history.append(
            {
                "question": result.question,
                "answer": result.answer_text,
                "passages": result.passages,
            }
        )
