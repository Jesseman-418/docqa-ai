"""QA chain: ties retrieval to answer generation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from app.qa.prompts import build_extractive_prompt, format_context, EXTRACTIVE_SYSTEM_PROMPT
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """Structured answer returned by the QA chain."""

    question: str
    answer_text: str
    sources: list[str]
    passages: list[dict]
    mode: str  # "extractive" or "generative"


class QAChain:
    """Orchestrates retrieval and answer generation.

    Supports two modes:
    - **extractive** (default): presents the most relevant passages directly.
    - **generative**: sends passages + question to an OpenAI LLM (requires API key).
    """

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        use_llm: bool = False,
    ) -> Answer:
        """Answer a question against the indexed documents.

        Args:
            question: The user's natural-language question.
            top_k: Override for number of passages to retrieve.
            use_llm: If True and OPENAI_API_KEY is set, use generative mode.

        Returns:
            An Answer object.
        """
        context = self.retriever.retrieve_with_context(question, top_k=top_k)
        results = context["results"]

        if not results:
            return Answer(
                question=question,
                answer_text="No relevant passages found in the uploaded documents. Try rephrasing your question or uploading more documents.",
                sources=[],
                passages=[],
                mode="extractive",
            )

        sources = context["sources"]

        if use_llm and os.environ.get("OPENAI_API_KEY"):
            return self._generative_answer(question, results, sources)

        return self._extractive_answer(question, results, sources)

    def _extractive_answer(self, question: str, results: list[dict], sources: list[str]) -> Answer:
        """Build an extractive answer from the top passages."""
        lines = ["**Based on your documents, here are the most relevant passages:**\n"]

        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "unknown")
            score = r["score"]
            text = r["text"]
            lines.append(f"**{i}. [{source}]** (relevance: {score:.2f})")
            lines.append(f"> {text}\n")

        answer_text = "\n".join(lines)

        return Answer(
            question=question,
            answer_text=answer_text,
            sources=sources,
            passages=results,
            mode="extractive",
        )

    def _generative_answer(self, question: str, results: list[dict], sources: list[str]) -> Answer:
        """Generate an answer using OpenAI's API with retrieved context."""
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = build_extractive_prompt(question, results)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": EXTRACTIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            answer_text = response.choices[0].message.content or "No response generated."

            return Answer(
                question=question,
                answer_text=answer_text,
                sources=sources,
                passages=results,
                mode="generative",
            )

        except Exception as e:
            logger.error("LLM generation failed: %s. Falling back to extractive.", e)
            return self._extractive_answer(question, results, sources)
