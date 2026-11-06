from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.faiss_store import FaissStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import Reranker
from src.agent.query_rewriter import QueryRewriter
from src.agent.self_improving_agent import (
    SelfImprovingRAGAgent
)
from src.generation.generator import Generator
from src.evaluation.rag_evaluator import RAGEvaluator
from src.memory.experience_memory import (
    ExperienceMemory
)


print("Loading document...")

documents = load_document(
    "data/raw/python_dummies.pdf"
)

chunks = chunk_documents(
    documents,
    chunk_size=800,
    overlap=150
)


print("Loading models...")

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(texts)

faiss_store = FaissStore(
    embeddings.shape[1]
)

faiss_store.add(
    embeddings,
    chunks
)

bm25_store = BM25Store(chunks)

hybrid_store = HybridRetriever(
    faiss_store,
    bm25_store,
    embedding_model
)

reranker = Reranker()

rewriter = QueryRewriter()

generator = Generator()

evaluator = RAGEvaluator()

memory = ExperienceMemory()


agent = SelfImprovingRAGAgent(
    faiss_store,
    bm25_store,
    hybrid_store,
    reranker,
    embedding_model,
    rewriter,
    generator,
    evaluator,
    memory
)


query = "What is the correct way to handle exceptions in Python?"

result = agent.run(query)

print("\nOriginal query:")
print(result["query"])

print("\nRewritten query:")
print(result["rewritten_query"])

print("\nMode:")
print(result["mode"])

print("\nInitial strategy:")
print(result["initial_strategy"])


if result["memory_details"]["neighbors"]:
    print("\nSimilar runtime experiences:")

    for neighbor in result[
        "memory_details"
    ]["neighbors"]:
        print(
            round(
                neighbor["similarity"],
                3
            ),
            "|",
            neighbor["best_strategy"],
            "|",
            neighbor["query"]
        )


if result["memory_details"]["scores"]:
    print("\nMemory strategy scores:")

    for strategy, score in result[
        "memory_details"
    ]["scores"].items():
        print(
            strategy,
            ":",
            round(score, 3)
        )


print("\nAttempts:")

for attempt in result["attempts"]:
    print(
        attempt["strategy"],
        "| Retrieval:",
        round(
            attempt[
                "retrieval_relevance"
            ],
            3
        ),
        "| Relevance:",
        round(
            attempt[
                "answer_relevance"
            ],
            3
        ),
        "| Faithfulness:",
        round(
            attempt["faithfulness"],
            3
        ),
        "| Score:",
        round(
            attempt["overall_score"],
            3
        ),
        "| Success:",
        attempt["success"]
    )


print("\nBest strategy:")
print(result["best_strategy"])


print("\nFinal answer:\n")
print(result["answer"])