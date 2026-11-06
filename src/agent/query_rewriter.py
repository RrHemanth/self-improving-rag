from langchain_core.prompts import ChatPromptTemplate
from src.llm.provider import create_chat_model
from src.llm.response import extract_text


class QueryRewriter:
    def __init__(
        self,
        model="qwen2.5:3b",
        provider="ollama",
        api_key=None,
    ):
        self.llm = create_chat_model(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
Rewrite the user's question into a clearer search query for document retrieval.

Preserve the original meaning.
Add useful technical terminology or synonyms when appropriate.
Do not answer the question.
Return only the rewritten query.
"""
            ),
            (
                "human",
                "{query}"
            )
        ])

        self.chain = self.prompt | self.llm

    def rewrite(self, query):
        response = self.chain.invoke({
            "query": query
        })

        rewritten = extract_text(response.content).strip()

        if not rewritten:
            return query

        return rewritten