import json
import re

from ollama import chat
from src.llm.provider import create_chat_model


class Generator:
    def __init__(
        self,
        model="qwen2.5:3b",
        provider="ollama",
        api_key=None,
    ):
        self.model = model
        self.provider = provider.strip().lower()

        if self.provider not in ("ollama", "openai", "anthropic", "gemini"):
            raise ValueError(f"Unsupported provider: {provider}")

        self.llm = None

        if self.provider != "ollama":
            self.llm = create_chat_model(
                provider=self.provider,
                model=model,
                api_key=api_key,
                temperature=0,
            )

    def _complete(self, prompt):
        if self.provider == "ollama":
            response = chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0},
            )
            return response.message.content

        response = self.llm.invoke([
            {"role": "user", "content": prompt}
        ])

        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "".join(
                part if isinstance(part, str)
                else part.get("text", "")
                for part in content
            )

        raise TypeError("Unexpected LLM response format.")

    def generate(self, query, retrieved_chunks):
        context = "\n\n".join(
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in retrieved_chunks
        )

        prompt = f"""
Answer the user's question using only the retrieved context.

Answer the question directly and clearly.
Do not add information that is not supported by the context.
If the context does not contain enough information, say so.

Context:
{context}

Question:
{query}
"""

        response_text = self._complete(prompt)

        return response_text

    def regenerate(
        self,
        query,
        retrieved_chunks,
        evaluation,
    ):
        context = "\n\n".join(
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in retrieved_chunks
        )

        unsupported = evaluation.get(
            "unsupported_claims",
            [],
        )

        relevance_feedback = evaluation.get(
            "relevance_reason",
            "",
        )

        prompt = f"""
Rewrite the answer using only the retrieved context.

The previous answer did not fully satisfy the evaluator.

Evaluator feedback:
{relevance_feedback}

Unsupported claims from the previous answer:
{unsupported}

Requirements:
- Answer the question directly.
- Use only facts supported by the context.
- Remove unsupported claims.
- Prefer a concise answer.
- Include examples only when supported by the context.

Context:
{context}

Question:
{query}
"""

        response_text = self._complete(prompt)

        return response_text

    def add_citations(
        self,
        answer,
        retrieved_chunks,
    ):
        if not answer or not retrieved_chunks:
            return answer

        sentences = self._split_sentences(answer)

        if not sentences:
            return answer

        references = self._build_citation_context(
            retrieved_chunks
        )

        numbered_sentences = "\n".join(
            f"{i}. {sentence}"
            for i, sentence in enumerate(
                sentences,
                start=1,
            )
        )

        prompt = f"""
Match each answer sentence to the references that directly support it.

Do not rewrite the answer.
Do not explain anything.
Only return JSON.

Use this format:

{{
    "1": [1],
    "2": [1, 3]
}}

If a sentence is not supported by any reference, return an empty list.

Answer sentences:
{numbered_sentences}

References:
{references}
"""

        response_text = self._complete(prompt)

        try:
            citation_map = json.loads(
                response_text
            )
        except json.JSONDecodeError:
            return answer

        cited_sentences = []
        used_sources = set()

        for i, sentence in enumerate(
            sentences,
            start=1,
        ):
            source_numbers = citation_map.get(
                str(i),
                [],
            )

            source_numbers = [
                number
                for number in source_numbers
                if (
                    isinstance(number, int)
                    and 1 <= number <= len(retrieved_chunks)
                )
            ]

            if source_numbers:
                citation_text = "".join(
                    f"[{number}]"
                    for number in source_numbers
                )

                sentence = self._attach_citation(
                    sentence,
                    citation_text,
                )

                used_sources.update(
                    source_numbers
                )

            cited_sentences.append(
                sentence
            )

        cited_answer = " ".join(
            cited_sentences
        )

        if not used_sources:
            return cited_answer

        source_lines = []

        for number in sorted(used_sources):
            chunk = retrieved_chunks[
                number - 1
            ]

            source = chunk["source"]
            page = chunk.get("page")

            if page is None:
                source_lines.append(
                    f"[{number}] {source}"
                )
            else:
                source_lines.append(
                    f"[{number}] {source}, p. {page}"
                )

        return (
            cited_answer
            + "\n\nSources:\n"
            + "\n".join(source_lines)
        )

    def _build_citation_context(
        self,
        retrieved_chunks,
    ):
        references = []

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            source = chunk["source"]
            page = chunk.get("page")

            if page is None:
                location = source
            else:
                location = f"{source}, p. {page}"

            references.append(
                f"[{i}] {location}\n{chunk['text']}"
            )

        return "\n\n".join(references)

    def _split_sentences(self, text):
        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                text.strip(),
            )
            if sentence.strip()
        ]

    def _attach_citation(
        self,
        sentence,
        citation,
    ):
        if sentence[-1] in ".!?":
            return (
                sentence[:-1]
                + " "
                + citation
                + sentence[-1]
            )

        return (
            sentence
            + " "
            + citation
        )