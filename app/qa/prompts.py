"""Prompt templates for QA chain."""

EXTRACTIVE_SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question using ONLY the provided context passages. If the context does not contain enough information, say so clearly.

Rules:
- Base your answer strictly on the provided passages.
- Cite the source document for each claim.
- If multiple passages are relevant, synthesise them into a coherent answer.
- If the passages do not address the question, respond: "I could not find relevant information in the uploaded documents."
"""

EXTRACTIVE_USER_TEMPLATE = """Context passages (ranked by relevance):

{context}

---

Question: {question}

Provide a clear, concise answer based on the passages above. Cite the source file for each key point."""

CONTEXT_PASSAGE_TEMPLATE = """[Passage {rank} | Source: {source} | Relevance: {score:.2f}]
{text}
"""


def format_context(results: list[dict]) -> str:
    """Format retrieval results into a context string for the prompt."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            CONTEXT_PASSAGE_TEMPLATE.format(
                rank=i,
                source=r["metadata"].get("source", "unknown"),
                score=r["score"],
                text=r["text"],
            )
        )
    return "\n".join(parts)


def build_extractive_prompt(question: str, results: list[dict]) -> str:
    """Build the full extractive QA prompt."""
    context = format_context(results)
    return EXTRACTIVE_USER_TEMPLATE.format(context=context, question=question)
