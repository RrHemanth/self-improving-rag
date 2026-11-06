from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


def create_chat_model(
    provider,
    model,
    api_key=None,
    **kwargs
):
    provider = provider.strip().lower()

    if not model or not model.strip():
        raise ValueError("A model name is required.")

    if provider == "ollama":
        return ChatOllama(
            model=model,
            **kwargs
        )

    if provider in ("openai", "anthropic", "gemini"):
        if not api_key or not api_key.strip():
            raise ValueError(
                f"An API key is required for {provider}."
            )

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            **kwargs
        )

    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            **kwargs
        )

    if provider == "gemini":
        kwargs.setdefault("timeout", 30)
        kwargs.setdefault("max_retries", 1)
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            **kwargs
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )
