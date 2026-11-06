import re

from ollama import chat


class RelevanceEvaluator:
    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def score(self, query, results):
        context = ""

        for i, result in enumerate(results, 1):
            context += (
                f"\nChunk {i} - Page {result['page']}:\n"
                f"{result['text']}\n"
            )

        prompt = f"""
You are evaluating the quality of retrieved passages for a RAG system.

Query:
{query}

Retrieved passages:
{context}

Judge how relevant the ENTIRE retrieved set is to the query.

Use this strict scale:

1 = Almost all retrieved passages are irrelevant.
2 = Only one or two passages are somewhat useful and most are irrelevant.
3 = Several passages are useful, but there is substantial irrelevant content.
4 = Most passages are directly relevant and useful for answering the query.
5 = Nearly every passage is highly relevant and directly helps answer the query.

Do not give a high score just because one passage contains the answer.
Penalize the retrieval set when several passages are unrelated to the query.

Return only one number from 1 to 5.
"""

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        output = response.message.content.strip()

        output = response.message.content.strip()

        print("\nRAW EVALUATOR OUTPUT:")
        print(output)

        match = re.search(r"\b[1-5]\b", output)

        if not match:
            return 0.0

        return float(match.group())