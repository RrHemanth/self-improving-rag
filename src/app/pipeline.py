import hashlib
from pathlib import Path

from src.ingestion.loader import load_documents
from src.ingestion.context_chunker import chunk_documents_contextual
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.faiss_store import FaissStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import Reranker
from src.agent.query_rewriter import QueryRewriter
from src.agent.self_improving_agent import SelfImprovingRAGAgent
from src.generation.generator import Generator
from src.evaluation.rag_evaluator import RAGEvaluator
from src.memory.experience_memory import ExperienceMemory


def build_corpus_id(file_paths):
    files = []

    for file_path in file_paths:
        path = Path(file_path)

        file_hash = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        files.append(
            f"{path.name}:{file_hash}"
        )

    value = "|".join(sorted(files))

    return hashlib.sha256(
        value.encode()
    ).hexdigest()[:12]


def build_agent(
    file_paths,
    provider="ollama",
    model=None,
    api_key=None,
):
    provider = provider.strip().lower()

    if provider not in ("ollama", "openai", "anthropic", "gemini"):
        raise ValueError(f"Unsupported provider: {provider}")

    if provider != "ollama":
        if not model or not model.strip():
            raise ValueError("Select a cloud model.")

        if not api_key or not api_key.strip():
            raise ValueError("An API key is required.")

    if provider == "ollama":
        rewriter_model = model or "qwen2.5:3b"
        evaluator_model = "qwen2.5:7b"
    else:
        rewriter_model = model
        evaluator_model = model

    documents = load_documents(file_paths)

    chunks = chunk_documents_contextual(
        documents,
        chunk_size=1000,
        overlap=150,
    )

    if not chunks:
        raise ValueError(
            "No usable text was found in the uploaded documents."
        )

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    faiss_store = FaissStore(
        embeddings.shape[1]
    )

    faiss_store.add(
        embeddings,
        chunks,
    )

    bm25_store = BM25Store(
        chunks
    )

    hybrid_store = HybridRetriever(
        faiss_store,
        bm25_store,
        embedding_model,
    )

    reranker = Reranker()
    rewriter = QueryRewriter(
        model=rewriter_model,
        provider=provider,
        api_key=api_key,
    )

    generator = Generator(
        model=rewriter_model,
        provider=provider,
        api_key=api_key,
    )

    evaluator = RAGEvaluator(
        model=evaluator_model,
        provider=provider,
        api_key=api_key,
    )

    corpus_id = build_corpus_id(
        file_paths
    )

    if provider == "ollama":
        configuration = f"{provider}:{rewriter_model}:eval:{evaluator_model}"
    else:
        configuration = f"{provider}:{model}"

    suffix = hashlib.sha256(
        configuration.encode()
    ).hexdigest()[:12]

    memory_path = f"data/memory/app/{corpus_id}_{suffix}.json"

    memory = ExperienceMemory(
        path=memory_path
    )

    agent = SelfImprovingRAGAgent(
        faiss_store,
        bm25_store,
        hybrid_store,
        reranker,
        embedding_model,
        rewriter,
        generator,
        evaluator,
        memory,
    )

    return {
        "agent": agent,
        "generator": generator,
        "chunks": chunks,
        "corpus_id": corpus_id,
        "documents": documents,
    }